import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import *

from grf.grf import Spectra
from grf.units import *

class DecaySpectra(Spectra):
    def __init__(self, cosmo=None, z_reio=7.82):
        """
        Class to compute CMB and 21cm spectra.
        :param cosmo: Cosmology as an astropy object
        :param z_reio: Redshift of reionization
        """
        Spectra.__init__(self, cosmo=cosmo, z_reio=z_reio)

        self.omega_21 = 5.9e-6 * eV  # Angular frequency of 21cm photons
        self.tau_u = 13.8e9 * Year  # Age of the Universe
        
        df_all = pd.read_csv('../data/apjlaabf86t2_ascii.txt', delimiter='\t',
                             skiprows=list(range(6)) + list(range(21, 24)),
                             names=['freq', 'survey', 'fid', 'unc', 'n/a'])
        df_arcade2 = df_all[df_all['survey'].str.contains("ARCADE")]
        df_llfss = df_all[df_all['survey'].str.contains("LLFSS")]
        df_radio = df_all[~df_all['survey'].str.contains("LLFSS|ARCADE")]

        self.omega_ARCADE2_ary = 2 * np.pi * df_arcade2['freq'].values * 1e9 / Sec
        self.T_ARCADE2_fid = df_arcade2['fid'].values
        self.T_ARCADE2_unc = df_arcade2['unc'].values

        self.omega_radio_ary = 2 * np.pi * df_radio['freq'].values * 1e9 / Sec
        self.T_radio_fid = df_radio['fid'].values
        self.T_radio_unc = df_radio['unc'].values

        self.omega_llfss_ary = 2 * np.pi * df_llfss['freq'].values * 1e9 / Sec
        self.T_llfss_fid = df_llfss['fid'].values
        self.T_llfss_unc = df_llfss['unc'].values

    def compute_temperature_evolution(self, m_a, tau_a_div_tau_u, z_ary, z_P_bins_ary, P_ary):
        """
        Computer 21cm temperature evolution, in standard scenario and with dark photon energy injection
        :param m_a: Mediator mass
        :param tau_a_div_tau_u: Mediator lifetime compared to lifetime of Universe
        :param z_ary: Redshifts at which to compute temperatures
        :param z_P_bins_ary: Redshift bins at which probabilities are specified
        :param P_ary: Probabilities corresponding to redshift bins `z_P_bins_ary`
        """

        # Lifetime
        self.tau_a = self.tau_u * tau_a_div_tau_u

        # Redshift bins at which transition probabilities specified
        z_P_bin_centers = 10 ** ((np.log10(z_P_bins_ary[1:]) + np.log10(z_P_bins_ary[:-1])) / 2.)

        # Differential number density (spectrum) of A' and CMB photons
        self.dn_SM_domega_ary = self.dn_CMB_domega(omega=self.omega_21, z=z_ary)
        self.dn_A_domega_ary = np.array([self.dn_A_domega(m_a=m_a, z=z, z_res_ary=z_P_bin_centers, P_ary=P_ary,
                                                          omega=self.omega_21, tau_a=self.tau_a) for z in
                                         z_ary])

        # Final temperature evolution
        self.T_CMB_SM = self.T_CMB_K(z_ary)  # Standard scenario
        self.T_CMB_A = self.T_CMB_K(z_ary) * (
                (self.dn_SM_domega_ary + self.dn_A_domega_ary) / self.dn_SM_domega_ary)  # With A' decays

    def T_CMB(self, z, T_0=None):
        """
        :param z: Redshift
        :return: CMB temperature in natural units
        """
        if T_0 is None:
            T_0_n = self.T_0_n
        else:
            T_0_n = (c.k_B * T_0 * u.Kelvin).to(u.eV).value * eV
        return T_0_n * (1 + z)

    def T_CMB_K(self, z):
        """
        :param z: Redshift
        :return: CMB temperature in K
        """
        return self.T_0 * (1 + z)

    def dn_A_domega(self, m_a, z, z_res_ary, P_ary, omega, tau_a):
        """
        :param m_a: Mass of DM
        :param z: Redshift at which spectrum of A is calculated
        :param z_res_ary: Array or redshifts over which transition probability specified
        :param P_ary: Transition probabilities in `z_res_ary`
        :param omega: Frequency at z
        :param tau_a: DM lifetime
        :return: Differential number density of photon A, dn/omega at frequency omega
        """
        omega_0 = omega / (1 + z)

        self.z_dec = m_a / 2 / omega_0 - 1
        self.t_c = self.cosmo.age(self.z_dec).value * 1e9 * Year
        self.H = self.cosmo.H(self.z_dec).value * Kmps / Mpc

        omega_res_ary = omega_0 * (1 + z_res_ary)

        dn_A_domega_ary = np.array(
            [self.dn_Ap_domega(m_a, z_res, omega_res, tau_a) * P * (
                    1 + z) ** 2 / (1 + z_res) ** 2 * np.heaviside(z_res - z, 0) for
             z_res, omega_res, P in zip(z_res_ary, omega_res_ary, P_ary)])

        return np.sum(dn_A_domega_ary)

    def dn_Ap_domega(self, m_a, z, omega, tau_a):
        """
        Differential number density of dark photon, from Eq. 22 of 1711.04531 and Eq. 7 of 1803.07048
        :param m_a: Mass of DM
        :param z: Redshift
        :param omega: Frequency in natural units
        :param tau_a: DM lifetime
        :return: Differential number density of dark photon A', dn/omega
        """

        omega_0 = omega / (1 + z)

        dn_Ap_domega_0 = 2 * self.cosmo.Odm0 * rho_c / (
                m_a * tau_a * omega_0 * self.H) * np.heaviside(self.z_dec, 0) * \
                         np.exp(-self.t_c / tau_a)

        return (1 + z) ** 2 * dn_Ap_domega_0 * np.heaviside(m_a / 2 / omega - 1, 0)

    def dn_CMB_domega(self, omega, z=0, T_0=None):
        """
        :param omega: Photon frequency
        :param z: Redshift
        :return: Number density spectrum of CMB
        """
        return 8 * np.pi * (omega / (2 * np.pi)) ** 2 * 1 / (np.exp(omega / self.T_CMB(z, T_0)) - 1) / (2 * np.pi)

    def B_CMB(self, omega, T):
        """ CMB spectral intensity
        """
        T_N = (c.k_B * T * u.Kelvin).to(u.eV).value * eV
        return omega ** 3 / (2 * np.pi ** 2) * (np.exp(omega / T_N) - 1) ** -1

    def f_a(self, tau_a, m_a):
        """
        From Eq. 5 of 1803.07048
        :param tau_a: Lifetime
        :param m_a: Dark matter mass
        :return: f_a
        """
        return np.sqrt(m_a ** 3 / (64 * np.pi * tau_a ** -1))

    def eps_stellar_cooling(self, f_a):
        """ Maximum allowed coupling from stellar cooling bounds, from 1803.07048
        """
        return 2e-9 * GeV ** -1 * f_a

    def tau_a_LL(self):
        """ Lower limit on mediator lifetime, from 1606.02073
        """
        return 1 / ((6.3e-3 / 1e9) * Year ** -1)

    def ratio_ARCADE2(self, m_a, z_bin_center_ary, P_ary, tau_a):

        omega_check_ary = np.concatenate([self.omega_radio_ary, self.omega_llfss_ary, [self.omega_ARCADE2_ary[0]]])
        T0_check_ary = np.concatenate([self.T_radio_fid + 2 * self.T_radio_unc, self.T_llfss_fid + 2 * self.T_llfss_unc,
                                       [(self.T_ARCADE2_fid + 2 * self.T_ARCADE2_unc)[0]]])

        dn_CMB_domega_ARCADE2_upper_check = self.dn_CMB_domega(omega_check_ary, z=0,
                                                               T_0=T0_check_ary)
        dn_A_domega_check = [self.dn_A_domega(m_a=m_a, z=0, z_res_ary=z_bin_center_ary, P_ary=P_ary,
                                              omega=omega, tau_a=tau_a) for omega in omega_check_ary]

        return np.min(dn_CMB_domega_ARCADE2_upper_check / dn_A_domega_check)
