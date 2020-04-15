import numpy as np
from scipy.interpolate import RegularGridInterpolator


class PowerSpectrumGridInterpolator():

    def __init__(self, ps_type='nonlin_matter', file_name=None, k_max=1e7):

        file_name = "../data/log_pk_grids/log_pk_" + ps_type + "_grid_ary.npz"
        if file_name is not None:
            file = np.load(file_name)

        log_pk_grid = file['log_pk_grid']
        z_grid = file['z_grid']
        k_grid = file['k_grid']

        self.z_min = np.min(z_grid)
        self.k_max = k_max

        self.interpolator = RegularGridInterpolator(points=[np.log10(z_grid), np.log10(k_grid)], values=log_pk_grid,
                                                    bounds_error=False,
                                                    fill_value=None)

    def __call__(self, z_ary, k_ary):

        log_z_mesh_ary, log_k_mesh_ary = np.log10(np.meshgrid(z_ary, k_ary))

        return np.where(np.transpose(log_z_mesh_ary) < np.log10(self.z_min),
                        self.interpolator(np.transpose(
                            [np.ones_like(log_z_mesh_ary) * np.log10(self.z_min), log_k_mesh_ary])),
                        self.interpolator(np.transpose([log_z_mesh_ary, log_k_mesh_ary])))
