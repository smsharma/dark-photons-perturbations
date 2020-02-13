import sys, os
import urllib.request
import itertools

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.misc import derivative
from scipy.stats import chi2
from scipy.integrate import quad
import scipy.linalg as la
from scipy.optimize import fsolve, minimize, brentq
from scipy.special import erf
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.io import fits
from classy import Class
from tqdm import *

from grf.units import *


class TransitionProbabilities:
    def __init__(self, cosmo=None, z_reio=7.82, z_recomb=1089.80):
        """
        Container class to compute dark photon transition probabilities.

        :param cosmo: Cosmology as an astropy object. If `None`, defaults to Planck18 cosmology.
        :param z_reio: Redshift of reionization. By default consistent with Planck18 cosmology.
        :param z_recomb: Redshift of recombination. By default consistent with Planck18 cosmology.
        """

        # Set constants
        self.xi_3 = 1.20205  # Apéry's constant, from Wikipedia
        self.eta = 6.129e-10  # Baryon-to-photon ratio, from 1912.01132
        self.Y_p = 0.247  # Helium-to-hydrogen mass fraction, from 1912.01132
        self.T_0 = 2.725  # CMB temperature today, from 0911.1955
        self.T_0_n = (c.k_B * self.T_0 * u.Kelvin).to(u.eV).value * eV  # CMB temperature today, in natural units

        # Set cosmology
        if cosmo is None:
            self.cosmo = self.get_Planck18_cosmology()
        else:
            self.cosmo = cosmo

        # Set redshift of reionization
        self.z_reio = z_reio
        self.z_recomb = z_recomb

        # Initialize CLASS instance to compute ionization fraction from
        self.initialize_class_inst()

    def get_Planck18_cosmology(self):
        """
        Stolen from https://github.com/abatten/fruitbat/blob/c074abff432c3b267d00fbb49781a0e0c6eeab75/fruitbat/cosmologies.py
        Planck 2018 paper VI Table 2 Final column (68% confidence interval)
        This is the Planck 2018 cosmology that will be added to Astropy when the
        paper is accepted.

        :return: astropy.cosmology.FlatLambdaCDM instance describing Planck18 cosmology
        """

        planck18_cosmology = {'Oc0': 0.2607,
                              'Ob0': 0.04897,
                              'Om0': 0.3111,
                              'H0': 67.66,
                              'n': 0.9665,
                              'sigma8': 0.8102,
                              'tau': 0.0561,
                              'z_reion': 7.82,
                              't0': 13.787,
                              'Tcmb0': 2.7255,
                              'Neff': 3.046,
                              'm_nu': [0., 0., 0.06],
                              'z_recomb' : 1089.80,
                              'reference': "Planck 2018 results. VI. Cosmological Parameters, "
                                           "A&A, submitted, Table 2 (TT, TE, EE + lowE + lensing + BAO)"
                              }

        Planck18 = FlatLambdaCDM(H0=planck18_cosmology['H0'],
                                 Om0=planck18_cosmology['Om0'],
                                 Tcmb0=planck18_cosmology['Tcmb0'],
                                 Neff=planck18_cosmology['Neff'],
                                 Ob0=planck18_cosmology['Ob0'], name="Planck18",
                                 m_nu=u.Quantity(planck18_cosmology['m_nu'], u.eV))

        return Planck18

    def initialize_class_inst(self):
        """ Get electron ionization fraction from CLASS
        """
        class_parameters = {'H0': self.cosmo.H0.value,
                            'Omega_b': self.cosmo.Ob0,
                            'N_ur': self.cosmo.Neff,
                            'Omega_cdm': self.cosmo.Odm0,
                            'YHe': self.Y_p,
                            'z_reio': self.z_reio}

        self.CLASS_inst = Class()
        self.CLASS_inst.set(class_parameters)
        self.CLASS_inst.compute()

        # z_ary = np.logspace(-3, 5, 100000)
        z_ary = np.linspace(0, 33000., 300000)
        x_e_ary = [self.CLASS_inst.ionization_fraction(z) for z in z_ary]
        self.x_e = interp1d(z_ary, x_e_ary)

    def m_A_sq(self, z, omega, x_e=None):
        """ Effective photon plasma mass squared, in natural units, from 1507.02614

            :param z: Redshift
            :param omega: Photon frequency
            :param x_e: Free electron fraction if not default (optional)
            :return: Effective photon plasma mass squared, in natural units
        """
        if x_e is None:
            x_e = self.x_e(z)
            
        m_A_sq = 1.4e-21 * (x_e - 7.3e-3 * (omega / eV) ** 2 * (1 - x_e)) * (
                self.n_p(z) / Centimeter ** -3)

        return m_A_sq * eV ** 2  # Convert to natural units

    def n_p(self, z):
        """ Proton density at redshift `z` in cm^3, from 1507.02614
        """
        return (1 - self.Y_p / 2.) * self.eta * 2 * self.xi_3 / np.pi ** 2 * (self.T_0_n * (1 + z)) ** 3

    def dz_dt(self, z):
        """ dz/dt
        """
        return - self.cosmo.H(z).value * Kmps / Mpc * (1 + z)

    def omega(self, omega_0, z, evolve_z=True):
        """ Frequency corresponding to present-day omega_0 evolved to `z` is `evolve_z` is `True`, otherwise
            just return `omega_0`.
        """
        if evolve_z:
            return omega_0 * (1 + z)
        else:
            return omega_0

    def get_z_crossings(self, m_A, omega_0, evolve_z=True):
        """
        Find redshifts at which resonance occurs

        :param m_A: Dark photon mass
        :param omega_0: Present-day frequency
        :param evolve_z: Whether to evolve frequency in redshift.
        :return: Array of redshifts at which resonance occurs
        """

        z_ary = np.logspace(-3, 4.5, 20000)

        m_A_ary = np.nan_to_num(np.sqrt(self.m_A_sq(z_ary, self.omega(omega_0, z_ary, evolve_z))), nan = 1e-18 * eV)
        
        where_ary = np.where(np.logical_or((m_A_ary[:-1] < m_A) * (m_A_ary[1:] > m_A), (m_A_ary[:-1] > m_A) * (m_A_ary[1:] < m_A)))

        m_A_sq = lambda z:  np.nan_to_num(np.sqrt(self.m_A_sq(z, self.omega(omega_0, z, evolve_z))), nan = 1e-18 * eV) - m_A 

        z_cross_ary = []
        for i in range(len(where_ary[0])):
            z_cross_ary.append(brentq(m_A_sq, z_ary[where_ary[0][i]], z_ary[where_ary[0][i] + 1]))
            
        return np.array(z_cross_ary)

    def P_trans(self, m_A, z_res_ary, omega_0, eps, evolve_z=True, approx_linearize=True):
        """
        Photon transition probability

        :param m_A: Dark photon mass
        :param z_res_ary: Array of resonance redshifts
        :param omega_0: Photon frequency (present day if `approx_linearize`, otherwise absolute)
        :param eps: Kinetic mixing coupling
        :param evolve_z: Whether to evolve `omega_0` in redshift
        :param approx_linearize: Linearize probability in `epsilon`
        :return: Transition probability array at redshifts `z_res_ary`
        """
    
        d_log_m_A_sq_dz = np.array(
            [derivative(lambda z: np.log(self.m_A_sq(z=z, omega=self.omega(omega_0, z, evolve_z))), x0=z, dx=1e-7) for z
            in z_res_ary])
        
        omega_res_ary = self.omega(omega_0, z_res_ary, evolve_z)

        if approx_linearize:
            P_homo = np.pi * m_A ** 2 * eps ** 2 / omega_res_ary * \
                     np.abs((d_log_m_A_sq_dz * self.dz_dt(z_res_ary))) ** -1
        else:
            r = np.abs((d_log_m_A_sq_dz * self.dz_dt(z_res_ary))) ** -1
            k = m_A ** 2 / (2 * omega_res_ary)
            P_homo = 1 - np.exp(-2 * np.pi * r * k * np.sin(eps) ** 2)

        return np.nan_to_num(P_homo)

    def P_tot(self, omega_0, eps, m_A, approx_linearize=True, evolve_z=True, sum_probs=False, **kwargs):
        """
        Total conversion probability in the homogeneous limit

        :param omega_0: Present-day photon frequency
        :param eps: Dark photon coupling
        :param m_A: Dark photon mass
        :param approx_linearize: Whether to use linearized probability approximation
        :param evolve_z: Whether to evolve frequency in redshift. 
        :param sum_probs: Whether to sum over probabilities associated with different z
        :return: Redshift resonance array, transition probability array
        """

        # Find redshift at which resonance occurs
        z_res_ary = [self.get_z_crossings(m_A, omega, evolve_z) for omega in omega_0]
        
        # Get transition probabilities at resonance

        if sum_probs:
            P_ary = np.array([np.nansum(self.P_trans(m_A, z, omega, eps, approx_linearize=approx_linearize, evolve_z=evolve_z)) for z,omega in zip(z_res_ary, omega_0)])
        else:
            P_ary = np.array([(self.P_trans(m_A, z, omega, eps, approx_linearize=approx_linearize, evolve_z=evolve_z)) for z,omega in zip(z_res_ary, omega_0)])
            
        return z_res_ary, 1., P_ary

    def B_CMB(self, omega, T):
        """ CMB spectral intensity at frequency `omega` (in natural units) for temperature `T` (in Kelvin)
        """
        T_N = (c.k_B * T * u.Kelvin).to(u.eV).value * eV
        return omega ** 3 / (2 * np.pi ** 2) * (np.exp(omega / T_N) - 1) ** -1


class PerturbedProbability(TransitionProbabilities):
    """ Class to compute photon-to-dark photon transition probabilities accounting for inhomogeneities
    """

    def __init__(self, log_pk_b_interp_fn, **kwargs):
        TransitionProbabilities.__init__(self, **kwargs)
        self.log_pk_b_interp_fn = log_pk_b_interp_fn

    def _delta_sq(self, z_ary, k_min, k_max, log_pk_interp_fn, r_smooth=None):
        """ Get squared variance of a power spectrum between scales k_min and k_max over redshifts z_ary. Returns matrix over redshifts.
        """
        k_ary = np.logspace(np.log10(k_min), np.log10(k_max), 100)
        integrand_ary = (k_ary ** 2 / (2 * np.pi ** 2)) * 10 ** log_pk_interp_fn(z_ary, k_ary) 

        if r_smooth is not None:
            integrand_ary *= self.tophat(r_smooth, k_ary)

        return np.trapz(integrand_ary, k_ary, axis=1)

    def tophat(self, r, k_ary):
        """ Top-hat filter of size `r` acting over Fourier-space frequency array `k_ary`
        """
        k_ary = np.array(k_ary)
        kr = k_ary * r
        w = 3 * (np.sin(kr) / kr **3 - np.cos(kr) / kr ** 2)
        w[k_ary == 0] = 1.0
        return w

    def _dP_dz(self, z_ary, m_Ap, k_min, k_max, omega, pdf='lognormal', one_plus_delta_bound=None, b=1, Ap_DM=False, x_e_ary=None, eng_weight=False, r_smooth=None, return_pdf=False):
        """ Differential photon-to-dark photon transition probability. Set eng_weight=True to get the dark photon-to-photon mean energy injection. 
        
        :param z_ary: Array of redshifts over which to calculate dP/dz
        :param m_Ap: Dark photon mass in natural units
        :param k_min: Minimum scale in h/Mpc
        :param k_max: Maximum scale in h/Mpc
        :param omega: Array of frequencies, in natural units
        :param pdf: Distribution of perturbations. 'gaussian', 'lognormal', 'voids', or a `RegularGridInterpolator` instance for a custom pdf.
        :param one_plus_delta_bound: Restrict to fluctuations (1 + \delta) times and divided by this quantity
        :param b: Bias parameter for lognormal pdf
        :param Ap_DM: Dark photon dark matter case
        :param x_e_ary: An optional custom ionization fraction array of same length at z_ary
        :param eng_weight: Weight dP/dz by m_Ap ** 2 / m_A_sq, for dark photon -> photon energy injection case. 
        :param r_smooth: Smoothing scale, by default no smoothing
        :param return_pdf: Whether to return pdf
        :return: dP_dz_ary, sigma_sq_ary (in natural units), (optionally) pdf_ary
        """ 

        z_mesh, omega_res_mesh = np.meshgrid(z_ary, omega)

        if not Ap_DM:
            omega_res_mesh *= (1 + z_mesh)

        # If using custom x_e

        if x_e_ary is None:   
            x_e_ary = self.x_e(z_mesh)
            m_A_sq  = self.m_A_sq(z_mesh, omega_res_mesh)

        else:
            x_e_mesh, _ = np.meshgrid(x_e_ary, omega)
            m_A_sq = self.m_A_sq(z_mesh, omega_res_mesh, x_e=x_e_mesh)


        if eng_weight:
            weight = m_Ap**2 / m_A_sq
        else:
            weight = 1.

        # Get variance of fluctuations

        delta_b_b = delta_e_e = delta_e_b = self._delta_sq(z_ary, k_min, k_max, self.log_pk_b_interp_fn, r_smooth=r_smooth)
        
        n_b_ary = self.n_p(z_mesh) / (Centimeter ** -3)
        A = 1.4e-21 * eV ** 2
        B = 1.02e-23
        
        sigma_sq_ary = n_b_ary ** 2 * (A ** 2 * x_e_ary ** 2 * delta_e_e - 2 * A * B * omega_res_mesh ** 2 * x_e_ary * (1 - x_e_ary) * delta_e_b + B ** 2 * omega_res_mesh ** 4 * delta_b_b  * (1 - x_e_ary) ** 2)

        one_plus_delta_ary = m_Ap ** 2 / m_A_sq

        # Different expressions for different pdf choices

        if pdf == 'lognormal':
            
            sigma_LN_sq_ary = np.log(1 + sigma_sq_ary / m_A_sq ** 2)

            pdf_ary = 1 / np.sqrt(2 * np.pi * sigma_LN_sq_ary) * np.exp(-(np.log(1 / b * (m_Ap ** 2 / m_A_sq + b - 1)) + sigma_LN_sq_ary / 2.) ** 2 / (2 * sigma_LN_sq_ary)) * (1. / (m_Ap ** 2 + (b - 1) * m_A_sq))

            dP_dz_ary = 1 / np.sqrt(2 * np.pi * sigma_LN_sq_ary) * np.exp(-(np.log(1 / b * (m_Ap ** 2 / m_A_sq + b - 1)) + sigma_LN_sq_ary / 2.) ** 2 / (2 * sigma_LN_sq_ary)) / np.abs(self.dz_dt(z_mesh)) * (m_Ap ** 2 / (m_Ap ** 2 + (b - 1) * m_A_sq))

            dP_dz_ary *= weight

            sigma_sq_ary = sigma_LN_sq_ary
     
        elif pdf == "gaussian":

            pdf_ary = 1. / np.sqrt(2 * np.pi * sigma_sq_ary) * np.exp(-(m_A_sq - m_Ap ** 2) ** 2 / (2 * sigma_sq_ary)) 
                    
            dP_dz_ary = m_Ap ** 2 / np.sqrt(2 * np.pi * sigma_sq_ary) * np.exp(-(m_A_sq - m_Ap ** 2) ** 2 / (2 * sigma_sq_ary)) / np.abs(self.dz_dt(z_mesh))

            dP_dz_ary *= weight

        elif pdf == 'voids':

            # Fluctuations PDF fits. 

            z_void_data_ary = np.array([0., 0.3, 0.6, 1., 1.5, 2.2, 3., 4., 5.3, 6.9, 9., 12])

            mu_void_data_ary = [
                -2.208, -2.147, -2.071, -1.991, -1.9093, -1.8296, -1.7554, 
                -1.6879, -1.5568, -1.5179, -1.4910, -1.4705
            ]
            sigma_void_data_ary = [
                0.260, 0.248, 0.226, 0.205, 0.1800, 0.1522, 
                0.1248, 0.0993, 0.0775, 0.0732, 0.0680, 0.0635
            ]
            alpha_void_data_ary = [
                2.600, 2.580, 2.415, 2.256, 2.0624, 1.8143, 
                1.4656, 1.0883, -0.7072, -1.1162, -1.4970, -1.9580
            ]

            mu_void_ary    = np.interp(z_mesh.flatten(), z_void_data_ary, mu_void_data_ary).reshape(z_mesh.shape)
            sigma_void_ary = np.interp(z_mesh.flatten(), z_void_data_ary, sigma_void_data_ary).reshape(z_mesh.shape)
            alpha_void_ary = np.interp(z_mesh.flatten(), z_void_data_ary, alpha_void_data_ary).reshape(z_mesh.shape)

            # Void Volume PDF fits. 

            vol_pow_data_ary = np.array([0.474, 0.459, 0.442, 0.423, 0.405, 0.385, 0.363, 0.343, 0.324, 0.309, 0.294, 0.283])

            V0_data_ary = np.array([717.1, 660.1, 598.3, 540.0, 487.6, 440.0, 399.0, 363.8, 334.9, 311.2, 290.9, 275.0])

            # This is the integral of V f(V) dV, mean volume of voids.  
            # Zero out if outside of range. 
            mean_vol_voids_ary = np.interp(z_mesh.flatten(), z_void_data_ary, V0_data_ary * (1. - vol_pow_data_ary), left=0., right=0.).reshape(z_mesh.shape)

            # Digitized (redshift, number of voids).
            counts_data = np.transpose([[0.00000000000000000, 35094.27929059312], [0.10499248627324143, 36325.89411654076], [0.25668117775442845, 37299.15681910792], [0.4746427908826387, 38418.173839450836],[0.652782922945601, 39499.57684230323], [0.8508446876311933, 40590.38334952826], [1.0688063007594053, 41709.400369871175], [1.306689546510249, 42837.820894586715], [1.5644975369094425, 43972.958208139855], [1.8622405973125318, 45047.644422154655], [2.2196863150691915, 46204.27545998809], [2.6370494199538452, 47257.467949722595], [3.0744311960811963, 48236.28726850994], [3.512035764958684, 49022.76217967532], [3.9496799414362, 49775.04252948568], [4.387571665413871, 50313.60687082719], [4.82551784984158, 50805.15369030555], [5.263473936169291, 51288.151869445166], [5.701548845297081, 51668.56636451972], [6.13965841107489, 52019.06061840864], [6.5777679768527015, 52369.55487229756], [7.015897346430524, 52702.95184550897], [7.454125735008409, 52950.86241533284], [7.8923689764363045, 53185.95002464858], [8.330602315964192, 53429.586274303074], [8.768845557392087, 53664.67388361881], [9.207118504520002, 53874.115571918286], [9.645450863047953, 54032.265418185234], [10.083783221575905, 54190.41526445219], [10.52210567820385, 54357.11375105789], [10.960438036731802, 54515.26359732485], [11.398770395259753, 54673.413443591795], [12.0000000000000, 54945.35519125683]])

            counts_ary = np.interp(z_mesh.flatten(), counts_data[0,:], counts_data[1,:]).reshape(z_mesh.shape)

            # Fraction of volume in voids. Simulation size: 500 h^-3 Mpc^3
            voids_vol_frac = mean_vol_voids_ary * counts_ary / 500.**3

            Omega_m = self.get_Planck18_cosmology().Om0

            t_ary = (np.log10(one_plus_delta_ary*Omega_m) - mu_void_ary) / sigma_void_ary

            voids_pdf = voids_vol_frac / np.sqrt(2*np.pi)*np.exp(-t_ary**2/2) * (1. + erf(alpha_void_ary*t_ary/np.sqrt(2))) / sigma_void_ary / one_plus_delta_ary / np.log(10)

            pdf_ary  = voids_pdf / m_A_sq

            dP_dz_ary = voids_pdf / np.abs(self.dz_dt(z_mesh)) * m_Ap**2 / m_A_sq

            dP_dz_ary *= weight

        else:
            
            # If using a custom pdf

            if np.shape(omega) == ():
                omega = np.array([omega])
            input_ary = np.zeros((len(omega), len(z_ary), 2))
            input_ary[:,:,1] =  one_plus_delta_ary
            input_ary[:,:,0] = [z_ary for _ in range(len(omega))]

            # Get PDF
            log_pdf = pdf(np.log10(input_ary))

            pdf_ary = 10 ** log_pdf / m_A_sq

            dP_dz_ary = (10 ** log_pdf) / np.abs(self.dz_dt(z_mesh)) * m_Ap ** 2 / m_A_sq

            dP_dz_ary *= weight

            # Switch to Gaussian above z = 200.

            pdf_gauss_ary = 1. / np.sqrt(2 * np.pi * sigma_sq_ary) * np.exp(-(m_A_sq - m_Ap ** 2) ** 2 / (2 * sigma_sq_ary))

            dP_dz_gauss_ary = m_Ap ** 2 / np.sqrt(2 * np.pi * sigma_sq_ary) * np.exp(-(m_A_sq - m_Ap ** 2) ** 2 / (2 * sigma_sq_ary)) / np.abs(self.dz_dt(z_mesh))

            dP_dz_gauss_ary *= weight

            pdf_ary[:, z_ary > 200.] = pdf_gauss_ary[:, z_ary > 200.]

            dP_dz_ary[:, z_ary > 200.] = dP_dz_gauss_ary[:, z_ary > 200.]

        # Remove small and large perturbation regions corresponding to `one_plus_delta_bound`
        if one_plus_delta_bound is not None:

            restrict_index = (one_plus_delta_ary > one_plus_delta_bound) + (one_plus_delta_ary < (1 / one_plus_delta_bound))
            pdf_ary[restrict_index] = 0.
            dP_dz_ary[restrict_index] = 0.

        dP_dz_ary *= np.pi * m_Ap ** 2 / omega_res_mesh

        if not return_pdf:

            return dP_dz_ary, sigma_sq_ary

        else: 

            return dP_dz_ary, sigma_sq_ary, pdf_ary
    
    def P_tot_perturb(self, omega, eps, m_Ap, k_min=1e-3, k_max=1e4, z_min=1e-3, z_max=1.5e2, z_excise_min=None, z_excise_max=None, z_int=None, pdf='lognormal', one_plus_delta_bound=None, b=1., Ap_DM=False, eng_weight=False, r_smooth=None, n_z_bins=10000):
        """
        Get total transition probability accounting for inhomogeneities.

        :param omega: Array of frequencies
        :param eps: Dark photon mixing parameter
        :param m_Ap: Dark photon mass in natural units
        :param k_min: Minimum scale in h/Mpc
        :param k_max: Maximum scale in h/Mpc
        :param z_min: Minimum redshift at which dP/dz will be calculated
        :param z_max: Maximum redshift at which dP/dz will be calculated. Above this, homogeneous approx will be used.
        :param z_excise_min: Minimum redshift which will be excised
        :param z_excise_max: Maximum redshift which will be excised
        :param z_int: 2-tuple with lower and upper redshift integration ranges
        :param pdf: Distribution of perturbations. 'gaussian', 'lognormal', 'voids', or a `RegularGridInterpolator` instance for a custom pdf.
        :param one_plus_delta_bound: Restrict to fluctuations (1 + \delta) times and divided by this quantity
        :param b: Bias parameter for lognormal pdf
        :param Ap_DM: Dark photon dark matter case
        :param eng_weight: Weight dP/dz by m_Ap**2/m_A_sq, for dark photon -> photon energy injection case. 
        :param r_smooth: Smoothing scale, by default no smoothing
        :param n_z_bins: Number of redshift bins between `z_min` and `z_max`
        :return: z_ary, dP_dz_ary, P_tot, sigma_sq_ary
        """
        if Ap_DM:
            evolve_z = False
        else:
            evolve_z = True

        # Redshifts array over which dP/dz will be calculated
        z_ary = np.logspace(np.log10(z_min), np.log10(z_max), n_z_bins)

        # Get homogeneous transitions
        z_cross_ary, _, P_tot_ary = self.P_tot(omega, eps, m_Ap, evolve_z=evolve_z)
        
        # Only keep homogeneous transitions above z_max
        P_tot_homo_list = [P_tot[z_cross > z_max] for P_tot, z_cross in zip(P_tot_ary, z_cross_ary)]

        z_cross_homo_list = [z_cross[z_cross > z_max] for z_cross in z_cross_ary]

        # Sum the homogeneous probabilities above z_max that we kept
        P_tot_homo_ary = np.array([np.sum(P_tot[z_cross > z_max]) for P_tot, z_cross in zip(P_tot_ary, z_cross_ary)])

        # If restricting integrations range
        if z_int is not None:

            # Restrict redshifts which will go into dP/dz to be in integration range
            z_ary = z_ary[(z_ary > z_int[0]) & (z_ary < z_int[1])] 

            # Restrict homogeneous probabilities within integration range and sum
            P_tot_homo_ary = np.array([np.sum(P_tot[(z_cross > z_int[0]) & (z_cross < z_int[1])]) for P_tot, z_cross in zip(P_tot_homo_list, z_cross_homo_list)])

        # Get differential probability
        dP_dz_ary, sigma_sq_ary = self._dP_dz(z_ary, m_Ap, k_min, k_max, omega, pdf, one_plus_delta_bound, b, Ap_DM=Ap_DM, eng_weight=eng_weight, r_smooth=r_smooth)

        dP_dz_ary *= eps ** 2

        # Excise a redshift range if needed
        if z_excise_min is not None and z_excise_max is not None:
            
            z_excised_low_ary = z_ary[z_ary < z_excise_min]
            z_excised_high_ary = z_ary[z_ary > z_excise_max]

            dP_dz_excised_low_ary = np.split(np.extract(np.tile(z_ary < z_excise_min, (len(omega),1)), dP_dz_ary), len(omega))
            dP_dz_excised_high_ary = np.split(np.extract(np.tile(z_ary > z_excise_max, (len(omega),1)), dP_dz_ary), len(omega))
            
            P_tot = np.trapz(np.nan_to_num(dP_dz_excised_low_ary), z_excised_low_ary) + np.trapz(np.nan_to_num(dP_dz_excised_high_ary), z_excised_high_ary)

        else:

            P_tot = np.trapz(np.nan_to_num(dP_dz_ary), z_ary)
        
        # Add homogeneous transitions to total probability array, unless dealing with voids
        if not pdf == "voids":
            P_tot += P_tot_homo_ary

        # Return arrays of interest
        return z_ary, dP_dz_ary, P_tot, sigma_sq_ary


class FIRAS(PerturbedProbability):
    """ Class to (down)load FIRAS data and perform fits including conversion of CMB photons to dark photons
    """

    def __init__(self, log_pk_b_interp_fn, PIXIE=False, **kwargs):
        PerturbedProbability.__init__(self, log_pk_b_interp_fn, **kwargs)                                 
        self.eps_base = 1e-7
        self.PIXIE = PIXIE
        self.set_up_firas()  # Set up FIRAS data etc

    def set_up_firas(self):
        """ Load FIRAS data (download if not available) and construct covariance matrices
        """

        # Download data if not already existing
        if not os.path.isfile('../data/firas_monopole_spec_v1.txt'):
            url_spec = 'https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt' 
            urllib.request.urlretrieve(url_spec, '../data/firas_monopole_spec_v1.txt')  

        df = pd.read_table('../data/firas_monopole_spec_v1.txt', 
            skiprows=18, sep='\s+', header=None, 
            names =['freq', 'I', 'residual', 'uncert', 'poles'])

        self.df = df
        
        # The FIRAS frequency points
        nu_minus_nu_prime_ary = np.linspace(2.27, 21.33, 43) - 2.27

        # Array to construct covariance matrix (from astro-ph/9605054)
        Q_ary = np.array([1.000, 0.176,-0.203, 0.145, 0.077,-0.005,-0.022, 0.032,
            0.053, 0.025,-0.003, 0.007, 0.029, 0.029, 0.003, -0.002, 0.016, 0.020, 0.011, 0.002, 0.007,
            0.011, 0.009, 0.003, -0.004, -0.001, 0.003, 0.003, -0.001,-0.003, 0.000, 0.003, 0.009, 0.015,
            0.008, 0.003,-0.002, 0.000, -0.006, -0.006, 0.000, 0.002, 0.008])

        nu = df['freq'].values

        # PIXIE parameters (following 1507.02614)
        if not self.PIXIE:

            self.omega_FIRAS = 2 * np.pi * nu * Centimeter ** -1
            self.d = df['I'].values
            unc = df['uncert'].values * 1e-3
            self.resid = df['residual'].values

            Cov = np.zeros((len(df), len(df)))
            for i1 in range(len(df)):
                for i2 in range(len(df)):
                    nu_minus_nu_prime = np.abs(nu[i1] - nu[i2])
                    idx_Q = (np.abs(nu_minus_nu_prime_ary - nu_minus_nu_prime)).argmin()
                    Q = Q_ary[idx_Q]
                    Cov[i1, i2] = unc[i1] * unc[i1] * Q

        else:
            
            nu_PIXIE = np.linspace(30 * 1e9 * Hz, 6 * 1e12 * Hz, 400) / (Centimeter ** -1)
            self.omega_FIRAS = 2 * np.pi * nu_PIXIE * Centimeter ** -1
            self.d = self.B_CMB(self.omega_FIRAS, 2.725) / (1e6 * Jy)
            unc = np.ones_like(nu_PIXIE) * 5 * Jy   / (1e6 * Jy)
            Cov = np.zeros((len(nu_PIXIE), len(nu_PIXIE)))
            for i1 in range(len(nu_PIXIE)):
                for i2 in range(len(nu_PIXIE)):
                    if i1 == i2:
                        Cov[i1, i2] = unc[i1] * unc[i1]
   
        self.Cinv = la.inv(Cov)

        # Null chi2
        self.chi2_null = minimize(self.chi2_FIRAS,x0=[2.725],args=(0, np.ones_like(self.omega_FIRAS)), method='Powell')

        # Find the critical value for 95% confidence for two-sided and one-sided chi2
        self.delta_chi2_95_level_one_sided = fsolve(lambda crit: (1. - chi2.cdf(crit, df=1.)) / 2. - 0.05, x0=[3.])[0]

        # Fiducial array of mixing angles to scan over
        self.eps_ary = np.logspace(-11., 0., 500)

    def chi2_FIRAS(self, x, eps, P_tot):
        """ The FIRAS \chi^2 statistic, to do minimization over blackbody temparature
        """
        T_0 = x[0]
        t = self.B_CMB(self.omega_FIRAS, T_0) * (1 - P_tot * (eps / self.eps_base) ** 2) / (1e6 * Jy)
        return (np.dot((self.d - t), np.matmul(self.Cinv, (self.d - t))))

    def chi2_FIRAS_scan(self, m_A, homo=False, **kwargs):
        """ Do a \chi^2 scan over the kinetic mixing parameter array defined in self.eps_ary

            :param m_A: Dark photon mass in natural units
            :param homo: Whether to use homogeneous limit
            :**kwargs: Keyword arguments passed to the transition probability function
            :return: \delta\chi^2 over the kinetic mixing parameter array defined in self.eps_ary
        """

        # Get total probability over the FIRAS frequencies
        if homo: 
            P_tot = self.P_tot(self.omega_FIRAS, self.eps_base, m_A, sum_probs=1, **kwargs)[2]  
        else:
            P_tot = self.P_tot_perturb(self.omega_FIRAS, self.eps_base, m_A, **kwargs)[2]          

        # Null chi2
        chi2_null = minimize(self.chi2_FIRAS,x0=[2.725],args=(0, np.ones_like(self.omega_FIRAS)), method='SLSQP').fun

        # Scan over epsilon

        chi2_ary = np.zeros(len(self.eps_ary))

        for i_eps, eps in enumerate(self.eps_ary):
            chi2_min = minimize(self.chi2_FIRAS, x0=[2.725],args=(eps, P_tot), method='SLSQP')
            chi2_ary[i_eps] = chi2_min.fun
        
        #Return \delta\chi^2
        return chi2_ary - chi2_null

    def get_lim(self, delta_chi2_ary):
        """ Get one-sided 95% containment limits. `delta_chi2_ary` must have shape (len(m_Ap_ary), len(self.eps_ary)).
            Returns limit array of length len(m_Ap_ary), over each mass index.
        """
    
        TS_m_xsec = np.zeros(3)

        TS_m_xsec[2] = self.eps_ary[0]

        lim_ary = np.zeros(len(delta_chi2_ary))

        for i_m in range(len(delta_chi2_ary)): 

            TS_eps_ary = np.nan_to_num(delta_chi2_ary[i_m], nan=1e10)

            # Find value, location and xsec at the max TS (as a function of mass)
            max_loc = np.argmin(TS_eps_ary)
            max_TS = TS_eps_ary[max_loc]

            if max_TS > TS_m_xsec[0]:
                TS_m_xsec[0] = max_TS
                TS_m_xsec[1] = i_m
                TS_m_xsec[2] = self.eps_ary[max_loc]

            # Calculate limit
            for xi in range(max_loc,len(self.eps_ary)):
                val = TS_eps_ary[xi]
                if val > self.delta_chi2_95_level_one_sided:
                    scale = (-TS_eps_ary[xi-1] + self.delta_chi2_95_level_one_sided) / (-TS_eps_ary[xi-1] + TS_eps_ary[xi])
                    lim_ary[i_m] = 10 ** (np.log10(self.eps_ary[xi - 1]) + scale * (np.log10(self.eps_ary[xi]) - np.log10(self.eps_ary[xi - 1])))
                    break  

        lim_ary[lim_ary == 0] = np.nan
        
        return lim_ary