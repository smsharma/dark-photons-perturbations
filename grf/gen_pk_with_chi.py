from classy import Class
import numpy as np
import argparse

output_dir = "/scratch/sm8383/pk_arys/"

parser = argparse.ArgumentParser()
parser.add_argument("--iz_compute", action="store",
                    dest="iz_compute", default=20, type=int)

results = parser.parse_args()
iz_compute = results.iz_compute

z_compute_ary = np.logspace(-3, 3, 500)

z_compute = z_compute_ary[iz_compute]

class_parameters = {'output': 'mTk,mPk',
                    'H0': 67.66,
                    'Omega_b': 0.04897,
                    'N_ur': 3.046,
                    'Omega_cdm': 0.2607,
                    'YHe': 0.245,
                    'z_reio': 7.82,
                    'n_s': 0.9665,
                    'A_s': 2.105e-9,
                    'P_k_max_1/Mpc': 500.0,
                    'perturbed recombination': 'y',
                    'non linear': 'halofit'
                    }

M = Class()
M.set(class_parameters)
M.set({'z_pk': z_compute})
M.compute()

h = M.h()  # get reduced Hubble for conversions to 1/Mpc

one_time = M.get_transfer(z_compute)

# Transfer functions

# Convert to units of Mpc^{-1}
k_ary = one_time['k (h/Mpc)'] * h

delta_b_ary = one_time['d_b']
delta_chi_ary = one_time['d_chi']

n_s = M.n_s()

# Primordial PS
k_pivot = 0.05
P_s = 2.105e-9 * (k_ary / k_pivot) ** (n_s - 1)

# Power spectra from transfer function
# In units of Mpc^3 / h^3
Pk_b_ary = P_s * (delta_b_ary) ** 2 / (k_ary ** 3 / (2 * np.pi ** 2)) * h ** 3
Pk_chi_ary = P_s * (delta_chi_ary) ** 2 / \
    (k_ary ** 3 / (2 * np.pi ** 2)) * h ** 3
Pk_chi_b_ary = P_s * (delta_chi_ary * delta_b_ary) / \
    (k_ary ** 3 / (2 * np.pi ** 2)) * h ** 3
Pk_tot_lin_ary = np.array([M.pk_lin(k, z_compute) * h ** 3 for k in k_ary])
Pk_tot_nonlin_ary = np.array([M.pk(k, z_compute) * h ** 3 for k in k_ary])

np.savez(output_dir + "p_k_chi_k_max_500_z_" + str(iz_compute),
         k_ary=k_ary / h,  # Convert back to h / Mpc
         Pk_b_ary=Pk_b_ary,
         Pk_chi_ary=Pk_chi_ary,
         Pk_chi_b_ary=Pk_chi_b_ary,
         Pk_e_ary=Pk_b_ary + Pk_chi_ary + 2 * Pk_chi_b_ary,
         Pk_e_b_ary=Pk_chi_b_ary + Pk_b_ary,
         Pk_tot_lin_ary=Pk_tot_lin_ary,
         Pk_tot_nonlin_ary=Pk_tot_nonlin_ary
         )
