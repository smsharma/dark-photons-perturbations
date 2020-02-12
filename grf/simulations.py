import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import *
from grf.grf import TransitionProbabilities

from classy import Class
from nbodykit.filters import TopHat
from nbodykit.lab import LinearMesh

from grf.units import *


class GaussianRandomFieldBoxes:
    def __init__(self, z_fid=20, z_range=[15, 25], k_max=0.1, n_points=100, r_filter=2., eps=1e-6, omega=5.9e-6 * eV, z_dep_P=True, cosmo=None, A_s=2.105e-9, use_nbodykit=True, generate_1d=False, log_pk_interp=None, seed=None):
        """
        :param z_fid: Fiducial redshift
        :param z_range: Range of redshifts to stack boxes over
        :param k_max: Maximum comoving scale in h/Mpc
        :param n_points: Number of points to simulate boxes with
        :param r_filter: Top-hat filter size relative to spacing size
        :param eps: Dark photon coupling \epsilon
        :param omega: Photon frequency in eV. By default = omega_21 ~ 5.9e-6 eV.
        :param z_dep_P: Whether power spectrum is varied by z
        :param cosmo: The cosmology specified as an astropy.cosmology.FlatLambdaCDM instance. Defaults to Planck18 cosmology.
        :param use_nbodykit: Whether to use nbodykit or home-grown code for GRF
        """

        self.z_fid = z_fid
        self.z_range = z_range
        self.k_max = k_max
        self.n_points = n_points
        self.r_filter = r_filter
        self.omega = omega
        self.eps = eps
        self.z_dep_P = z_dep_P
        self.A_s = A_s
        self.Y_p = 0.25  # Helium-to-hydrogen mass fraction, from 0901.0014

        # Set cosmology
        if cosmo is None:
            self.cosmo = self.get_Planck18_cosmology()
        else:
            self.cosmo = cosmo

        self.use_nbodykit = use_nbodykit
        self.generate_1d = generate_1d

        # If generating in 1-D need to use home-grown code
        if self.generate_1d:
            self.use_nbodykit = False

        self.log_pk_interp = log_pk_interp

        if seed is None:
            self.seed = np.random.randint(1, 1e8)
        else:
            self.seed = seed

        np.random.seed(self.seed)
        
        self.get_box_properties()
        self.simulate_grf()
        self.calculate_transition_prob()


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

    def get_box_properties(self):
        """ Assign properties to simulation boxes
        """

        # Total comoving distance between requested redshift range
        self.d_comoving_total = (self.cosmo.comoving_distance(self.z_range[1]) -
                                 self.cosmo.comoving_distance(self.z_range[0])).value / (1 / self.cosmo.h)

        # Number of points needed to get down to k_max resolution
        self.n_points_needed = self.k_max * self.d_comoving_total / (2 * np.pi)

        # Ratio of number of points needed and how many we're willing to run with
        self.ratio_n_points = self.n_points_needed / self.n_points

        # Number of boxes to simulate
        self.n_boxes = int(np.ceil(self.ratio_n_points))

        # Redshift bin edges and centers of boxes to simulate
        self.z_bins = np.linspace(self.z_range[0], self.z_range[1], self.n_boxes + 1)

        # If power spectrum a function of z grab those redshifts, otherwise only calculate at fiducial redshift
        if self.z_dep_P:
            self.z_center_ary = (self.z_bins[1:] + self.z_bins[:-1]) / 2.
        else:
            self.z_center_ary = np.array([self.z_fid])

        # Comoving distance at redshift bin edges, size of each box and number of points to use to get resolution k_max
        self.d_comoving_bins = self.cosmo.comoving_distance(self.z_bins).value / (1 / self.cosmo.h)
        self.delta_d_comoving_ary = self.d_comoving_bins[1:] - self.d_comoving_bins[:-1]
        self.n_points_ary = np.array(np.ceil(self.k_max * self.delta_d_comoving_ary / (2 * np.pi) / 2 ) * 2, dtype=np.int32)

    def simulate_grf(self):
        """ Simulate Gaussian random field and get 1-D (1 + \delta) slices through boxes
        """

        self.one_plus_delta_1d = []  # 1-D slice through 1 + \delta
        self.z_ary = []  # Redshift array corresponding to 1 + \delta

        if self.n_boxes < 10: disable_tqdm = True
        else: disable_tqdm = False

        self.fields = []

        for z_center, i_b in tqdm_notebook(zip(self.z_center_ary, range(self.n_boxes)), total=len(self.z_center_ary), disable=disable_tqdm):
            if self.z_dep_P:
                pk_interp_b = lambda k: 10 ** self.log_pk_interp(z_center, k)  # Interpolated baryon P(k)
            else:
                pk_interp_b = lambda k: 10 ** self.log_pk_interp(np.mean(self.z_range), k)  # Interpolated baryon P(k)
            spacing_size = self.delta_d_comoving_ary[i_b] / self.n_points_ary[i_b]  # Real-space spacing

            if self.use_nbodykit:
                mesh = LinearMesh(pk_interp_b, Nmesh=self.n_points_ary[i_b], BoxSize=self.delta_d_comoving_ary[i_b], seed=seed)
                
                # Apply tophat filter if needed
                if self.r_filter == 0:
                    mesh_smooth = mesh
                    self.filter_size = 0.
                else:
                    self.filter_size = self.r_filter * spacing_size
                    mesh_smooth = mesh.apply(TopHat(self.filter_size)).paint()
                print(self.filter_size)
                self.field = mesh_smooth.preview()
                self.fields.append(self.field)
                self.one_plus_delta_1d += [list(self.field[0, int(np.round(self.n_points_ary[i_b] / 2)), :])]
            else:
                if not self.generate_1d:
                    shape = np.array(3 * [self.n_points_ary[i_b]])
                    delta = self.generate_field(self.distrib, lambda k: pk_interp_b(k), shape, unit_length=spacing_size, filt_size=self.r_filter * spacing_size)
                    field = 1 + delta
                    self.one_plus_delta_1d += [list(field[0, int(np.round(self.n_points_ary[i_b] / 2)), :])]
                else:
                    shape = np.array([self.n_points_ary[i_b]])
                    delta = self.generate_field(self.distrib, lambda k: k ** 2 * pk_interp_b(k) / (2 * np.pi), shape, unit_length=spacing_size, filt_size=self.r_filter * spacing_size)
                    field = 1 + delta
                    self.one_plus_delta_1d += [list(field)]

            
            d_comov_ary = np.linspace(self.d_comoving_bins[i_b], self.d_comoving_bins[i_b + 1], self.n_points_ary[i_b])
            self.z_ary += [[z_at_value(self.cosmo.comoving_distance, d / self.cosmo.h * u.Mpc, zmax=1e5) for d in d_comov_ary]]

    def generate_field(self, statistic, power_spectrum, shape, unit_length=1, filt_size=0):
        """
        Generates a field given a statistic and a power_spectrum.
        Parameters
        ----------
        statistic: callable
            A function that takes returns a random array of a given signature,
            with signature (s) -> (B) with B.shape == s. Please note that the
            distribution is in *Fourier space* not in real space, unless you set
            stat_real=True. See the note below for more details.
        power_spectrum: callable
            A function that returns the power contained in a given mode,
            with signature (k) -> P(k) with k.shape == (ndim, n)
        shape: tuple
            The shape of the output field
        unit_length: float, optional
            How much physical length represent 1pixel. For example a value of 10
            mean that each pixel stands for 10 physical units. It has the
            dimension of a physical_unit/pixel.
        Returns
        -------
        field: a real array of shape `shape` following the statistic
            with the given power_spectrum
        Note
        ----
        When generation the distribution in Fourier mode, the result
        should be complex and unitary. Only the phase is random.
        """
        
        # Compute the k grid
        all_k = [np.fft.fftfreq(s, d=unit_length) for s in shape[:-1]] + \
                [np.fft.rfftfreq(shape[-1], d=unit_length)]

        kgrid = np.meshgrid(*all_k, indexing='ij')
        knorm = 2 * np.pi * np.sqrt(np.sum(np.power(kgrid, 2), axis=0))
            
        fourier_shape = knorm.shape
        
        vol = np.prod(np.array(shape) * unit_length)
        scale_factor = vol

        # Draw a random sample in Fourier space
        fftfield = statistic(fourier_shape)

        power_k = np.where(knorm == 0, 0, np.sqrt(power_spectrum(knorm) / scale_factor))
        
        fftfield *= power_k[0]

        if filt_size != 0:
            tophat_k = np.where(knorm == 0, 0, self.tophat(filt_size, knorm))
            fftfield *= tophat_k

        return np.prod(shape) * np.fft.irfftn(fftfield)

    def distrib(self, shape):
        # Build a unit-distribution of complex numbers with random phase
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return (a + 1.j * b)/np.sqrt(2)

    def distrib_real(self, shape):
        # Build a unit-distribution of complex numbers with random phase
        a = np.random.normal(loc=0, scale=1, size=shape)
        return a
        
    def tophat(self, r, knorm):
        knorm = np.array(knorm)
        kr = knorm * r
        w = 3 * (np.sin(kr) / kr **3 - np.cos(kr) / kr ** 2)
        w[knorm == 0] = 1.0
        return w

    def calculate_transition_prob(self):
        """ Calculate A' -> A transition probabilities
        """

        # Initialize transition probabilities class
        AO = AnisotropicOscillations()
        self.z_crossings_collect_temp = []
        self.P_collect_temp = []
        self.z_ary_new = []
        self.m_A_perturb_ary = []
        self.m_A_sq_ary = []

        # Loop over boxes and get crossings and transition probabilities

        for i_b in range(self.n_boxes):
            AO.get_m_A_perturb(z_fid=self.z_fid, omega=self.omega, z_ary=self.z_ary[i_b],
                               one_plus_delta_ary=np.nan_to_num(self.one_plus_delta_1d[i_b]))
            AO.get_crossings_and_derivatives()
            AO.get_P_ary(eps=self.eps, omega=self.omega)
            self.z_crossings_collect_temp += list(AO.z_crossings_ary)
            self.P_collect_temp += list(AO.P_ary)
            self.m_A_perturb_ary += list(AO.m_A_perturb_ary)
            self.m_A_sq_ary += list(AO.m_A_sq_ary)
            self.z_ary_new += list(AO.z_ary_new)

        self.P_homo = AO.P_homo
        self.m_A_fid = AO.m_A_fid


class AnisotropicOscillations(TransitionProbabilities):
    def __init__(self, cosmo=None, z_reio=7.82):
        TransitionProbabilities.__init__(self, cosmo=cosmo, z_reio=z_reio)

    def get_m_A_perturb(self, z_fid, omega, z_ary, one_plus_delta_ary):
        """ Get perturbations in plasma mass
        """

        self.z_ary_new = z_ary
        self.one_plus_delta_ary = one_plus_delta_ary
        self.m_A_fid = np.sqrt(self.m_A_sq(z_fid, omega))
        self.m_A_sq_ary = np.array([(self.m_A_sq(z, omega)) for z in self.z_ary_new])
        self.m_A_ary = np.sqrt(self.m_A_sq_ary)
        self.m_A_perturb_ary = self.m_A_ary * np.sqrt(np.nan_to_num(self.one_plus_delta_ary))

    def get_crossings_and_derivatives(self):
        """ Locate redshift of crossings and derivatives at crossings
        """

        z_crossings_ary = []
        d_log_m_A_sq_dz_ary = []
        dz = self.z_ary_new[1] - self.z_ary_new[0]

        for iz, z in enumerate(self.z_ary_new[:-1]):
            if (self.m_A_perturb_ary[iz] < self.m_A_fid and self.m_A_perturb_ary[iz + 1] > self.m_A_fid) \
                    or (self.m_A_perturb_ary[iz] > self.m_A_fid and self.m_A_perturb_ary[iz + 1] < self.m_A_fid):
                z_crossings_ary.append(np.interp(self.m_A_fid, [self.m_A_perturb_ary[iz], self.m_A_perturb_ary[iz + 1]],
                                                 [self.z_ary_new[iz], self.z_ary_new[iz + 1]]))
                d_log_m_A_sq_dz_ary.append(
                    (np.log(self.m_A_perturb_ary[iz + 1] ** 2) - np.log(self.m_A_perturb_ary[iz] ** 2)) / dz)

        self.d_log_m_A_sq_dz_homo = None

        for iz, z in enumerate(self.z_ary_new[:-1]):
            if (self.m_A_ary[iz] < self.m_A_fid and self.m_A_ary[iz + 1] > self.m_A_fid) \
                    or (self.m_A_ary[iz] > self.m_A_fid and self.m_A_ary[iz + 1] < self.m_A_fid):
                self.z_crossing_homo = np.interp(self.m_A_fid, [self.m_A_ary[iz], self.m_A_ary[iz + 1]],
                                                 [self.z_ary_new[iz], self.z_ary_new[iz + 1]])
                self.d_log_m_A_sq_dz_homo = (np.log(self.m_A_ary[iz + 1] ** 2) - np.log(self.m_A_ary[iz] ** 2)) / dz

                continue

        self.z_crossings_ary = np.array(z_crossings_ary)
        self.d_log_m_A_sq_dz_ary = np.array(d_log_m_A_sq_dz_ary)

    def get_P_ary(self, eps, omega):
        """ Get transition probabilities
        """

        # Transition probabilities
        self.P_ary = np.pi * self.m_A_fid ** 2 * eps ** 2 / omega * \
                     np.abs((self.d_log_m_A_sq_dz_ary * self.dz_dt(self.z_crossings_ary)) ** -1)

        # If range corresponds to homogeneous crossing, get that probability
        self.P_homo = None
        if self.d_log_m_A_sq_dz_homo is not None:
            self.P_homo = np.pi * self.m_A_fid ** 2 * eps ** 2 / omega * \
                          np.abs((self.d_log_m_A_sq_dz_homo * self.dz_dt(self.z_crossing_homo)) ** -1)
