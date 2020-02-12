
import sys, os
import random
import numpy as np
sys.path.append("../")
from grf.units import *

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu


# cd /tigress/smsharma/spectral-distortions-perturbations/cluster
cd /home/sm8383/spectral_distortions_perturbations/cluster

conda activate

'''

# for pdf in ["gaussian", "lognormal", "log_analytic_pdf_interp"]:
#     for pspec_tag in ["franken_lower", "franken_upper"]:
#         for one_plus_delta_bound in [-1, 1e2, 1e4]:
#             for z_min in [1e-3, 20]:
#                 for z_excise_max in [20, 30]:
#                     for b in [1, 1.5]:
#                         for pixie in [0, 1]:

#                             if (b != 1) and pdf is not "lognormal":
#                                 continue

#                             tag = "pdf_" + pdf + "_" + pspec_tag + "_cut_" + str(one_plus_delta_bound) + "_pixie_" + str(pixie) +'_z_min_' + str(z_min) + '_z_excise_max_' + str(z_excise_max) 

#                             if pdf is "lognormal":
#                                 tag += '_b_' + str(b)

#                             batchn = batch  + "\n"
#                             batchn += "python firas_interface.py --pdf " + pdf + " --tag " + tag + " --pspec_tag " + pspec_tag + " --one_plus_delta_bound " + str(one_plus_delta_bound) + " --pixie " + str(pixie) + " --z_min " + str(z_min) + " --z_excise_max " + str(z_excise_max) + " --b " + str(b)
#                             fname = "batch/submit.batch" 
#                             f=open(fname, "w")
#                             f.write(batchn)
#                             f.close()
#                             os.system("chmod +x " + fname)
#                             os.system("sbatch " + fname)

# for pixie in [0, 1]:

#     tag = "homo_pixie_" + str(pixie) 
#     batchn = batch  + "\n"
#     batchn += "python firas_interface.py --homo 1 --tag " + tag + " --pixie " + str(pixie)
#     fname = "batch/submit.batch" 
#     f=open(fname, "w")
#     f.write(batchn)
#     f.close()
#     os.system("chmod +x " + fname)
#     os.system("sbatch " + fname)

# for pdf in ["voids"]:
#     for pspec_tag in ["franken_lower"]:
#         for one_plus_delta_bound in [-1, 1e2]:
#             for z_min in [1e-3, 20]:
#                 for z_excise_max in [20, 30]:
#                     for b in [1]:
#                         for pixie in [0]:

#                             if (b != 1) and pdf is not "lognormal":
#                                 continue

#                             tag = "pdf_" + pdf + "_" + pspec_tag + "_cut_" + str(one_plus_delta_bound) + "_pixie_" + str(pixie) +'_z_min_' + str(z_min) + '_z_excise_max_' + str(z_excise_max) 

#                             if pdf is "lognormal":
#                                 tag += '_b_' + str(b)
                                
#                             batchn = batch  + "\n"
#                             batchn += "python firas_interface.py --pdf " + pdf + " --tag " + tag + " --pspec_tag " + pspec_tag + " --one_plus_delta_bound " + str(one_plus_delta_bound) + " --pixie " + str(pixie) + " --z_min " + str(z_min) + " --z_excise_max " + str(z_excise_max) + " --b " + str(b)
#                             fname = "batch/submit.batch" 
#                             f=open(fname, "w")
#                             f.write(batchn)
#                             f.close()
#                             os.system("chmod +x " + fname)
#                             os.system("sbatch " + fname)

# # Run more 1 + delta

# for pdf in ["lognormal", "log_analytic_pdf_interp"]:
#     for pspec_tag in ["franken_lower", "franken_upper"]:
#         for one_plus_delta_bound in [10, 1e3, 1e5]:
#             for z_min in [1e-3]:
#                 for z_excise_max in [20]:
#                     for b in [1]:
#                         for pixie in [0]:

#                             if (b != 1) and pdf is not "lognormal":
#                                 continue
#                             if (pspec_tag == "franken_upper") and (pdf == "log_analytic_pdf_interp"):
#                                 continue

#                             tag = "pdf_" + pdf + "_" + pspec_tag + "_cut_" + str(one_plus_delta_bound) + "_pixie_" + str(pixie) +'_z_min_' + str(z_min) + '_z_excise_max_' + str(z_excise_max) 

#                             if pdf is "lognormal":
#                                 tag += '_b_' + str(b)

#                             batchn = batch  + "\n"
#                             batchn += "python firas_interface.py --pdf " + pdf + " --tag " + tag + " --pspec_tag " + pspec_tag + " --one_plus_delta_bound " + str(one_plus_delta_bound) + " --pixie " + str(pixie) + " --z_min " + str(z_min) + " --z_excise_max " + str(z_excise_max) + " --b " + str(b)
#                             fname = "batch/submit.batch" 
#                             f=open(fname, "w")
#                             f.write(batchn)
#                             f.close()
#                             os.system("chmod +x " + fname)
#                             os.system("sbatch " + fname)

# Run more z_min

# for pdf in ["lognormal"]:
#     for pspec_tag in ["franken_lower", "franken_upper"]:
#         for one_plus_delta_bound in [1e2]:
#             for z_min in [1e-3, 1e-2, 1e-1, 1]:
#                 for z_excise_max in [20]:
#                     for b in [1]:
#                         for pixie in [0]:
#                             if (b != 1) and pdf is not "lognormal":
#                                 continue
#                             if (pspec_tag == "franken_upper") and (pdf == "log_analytic_pdf_interp"):
#                                 continue

#                             tag = "pdf_" + pdf + "_" + pspec_tag + "_cut_" + str(one_plus_delta_bound) + "_pixie_" + str(pixie) +'_z_min_' + str(z_min) + '_z_excise_max_' + str(z_excise_max) 

#                             if pdf is "lognormal":
#                                 tag += '_b_' + str(b)

#                             batchn = batch  + "\n"
#                             batchn += "python firas_interface.py --pdf " + pdf + " --tag " + tag + " --pspec_tag " + pspec_tag + " --one_plus_delta_bound " + str(one_plus_delta_bound) + " --pixie " + str(pixie) + " --z_min " + str(z_min) + " --z_excise_max " + str(z_excise_max) + " --b " + str(b)
#                             fname = "batch/submit.batch" 
#                             f=open(fname, "w")
#                             f.write(batchn)
#                             f.close()
#                             os.system("chmod +x " + fname)
#                             os.system("sbatch " + fname)

for pdf in ["lognormal"]:
    for pspec_tag in ["franken_lower", "franken_upper"]:
        for one_plus_delta_bound in [1e2]:
            for z_min in [1e-3, 20]:
                for z_excise_max in [20]:
                    for b in [1]:
                        for pixie in [0, 1]:

                            if (b != 1) and pdf is not "lognormal":
                                continue

                            tag = "widemass_1_pdf_" + pdf + "_" + pspec_tag + "_cut_" + str(one_plus_delta_bound) + "_pixie_" + str(pixie) +'_z_min_' + str(z_min) + '_z_excise_max_' + str(z_excise_max) 

                            if pdf is "lognormal":
                                tag += '_b_' + str(b)

                            batchn = batch  + "\n"
                            batchn += "python firas_interface.py --widemass 1 --pdf " + pdf + " --tag " + tag + " --pspec_tag " + pspec_tag + " --one_plus_delta_bound " + str(one_plus_delta_bound) + " --pixie " + str(pixie) + " --z_min " + str(z_min) + " --z_excise_max " + str(z_excise_max) + " --b " + str(b)
                            fname = "batch/submit.batch" 
                            f=open(fname, "w")
                            f.write(batchn)
                            f.close()
                            os.system("chmod +x " + fname)
                            os.system("sbatch " + fname)