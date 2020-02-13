import sys
sys.path.append("../")
import argparse
import pickle 

import numpy as np
from tqdm import *

from grf.grf import FIRAS
from grf.pk_interp import PowerSpectrumGridInterpolator
from grf.units import *

output_dir = "/scratch/sm8383/chi2_arys/firas_dp_scan_"
# output_dir = "/tigress/smsharma/chi2_arys/firas_dp_scan_"

parser = argparse.ArgumentParser()
parser.add_argument("--tag", action="store", dest="tag", default="test", type=str)
parser.add_argument("--pdf", action="store", dest="pdf", default="lognormal", type=str)
parser.add_argument("--pspec_tag", action="store", dest="pspec_tag", default="nonlin_matter", type=str)
parser.add_argument("--one_plus_delta_bound", action="store", dest="one_plus_delta_bound", default=1e2, type=float)

parser.add_argument("--b", action="store", dest="b", default=1., type=float)

parser.add_argument("--z_min", action="store", dest="z_min", default=1e-3, type=float)
parser.add_argument("--k_max", action="store", dest="k_max", default=1e4, type=float)

parser.add_argument("--z_excise_min", action="store", dest="z_excise_min", default=6, type=float)
parser.add_argument("--z_excise_max", action="store", dest="z_excise_max", default=20, type=float)

parser.add_argument("--pixie", action="store", dest="pixie", default=0, type=int)
parser.add_argument("--homo", action="store", dest="homo", default=0, type=int)

parser.add_argument("--widemass", action="store", dest="widemass", default=0, type=int)

results=parser.parse_args()


tag = results.tag
pdf = results.pdf
pspec_tag = results.pspec_tag
one_plus_delta_bound = results.one_plus_delta_bound

b = results.b

z_min = results.z_min
k_max = results.k_max

z_excise_min = results.z_excise_min
z_excise_max = results.z_excise_max

pixie = results.pixie
homo = results.homo

widemass = results.widemass

z_int = [z_min, 1e6]

if z_excise_min == -1:
    z_excise_min = None

if z_excise_max == -1:
    z_excise_max = None

if one_plus_delta_bound == -1:
    one_plus_delta_bound = None

if pdf not in ["gaussian", "lognormal", "voids"]:
    pdf = pickle.load(open('../data/analytic_pdf_grids/' + pdf + '.npy', 'rb'))

if not widemass:
    m_A_ary = np.logspace(-17, np.log10(1e-9), 500) * eV 
else:
    m_A_ary = np.logspace(-17, np.log10(1e-7), 1000) * eV 

pspec = PowerSpectrumGridInterpolator(pspec_tag)
firas = FIRAS(pspec, PIXIE=pixie)

delta_chi2_ary = np.zeros((len(m_A_ary), len(firas.eps_ary)))

for i_m_A, m_A in enumerate(tqdm(m_A_ary)):
    delta_chi2_ary[i_m_A, :] = firas.chi2_FIRAS_scan(m_A, k_max=k_max, z_excise_min=z_excise_min, z_excise_max=z_excise_max, homo=homo, one_plus_delta_bound=one_plus_delta_bound, z_int=z_int, pdf=pdf, b=b)

limit = np.nan_to_num(firas.get_lim(delta_chi2_ary), nan=2.)

np.savez(output_dir + tag, 
        delta_chi2_ary=delta_chi2_ary,
        limit=limit,
        m_A_ary=m_A_ary)
