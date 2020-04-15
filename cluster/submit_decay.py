import sys
sys.path.append("../")

import os
import random
import numpy as np

from grf.units import *

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu

cd /home/sm8383/spectral_distortions_perturbations/cluster

conda activate

'''

# Main set of plots

i_m_Ap_ary = np.arange(100)

for one_plus_delta_bound in [10, 100]:
    for k_max in [10, 1000]:
        for i_m_Ap in i_m_Ap_ary:
            batchn = batch + "\n"
            batchn += "python decay_interface.py --i_m_Ap " + \
                str(i_m_Ap) + " --one_plus_delta_bound " + \
                str(one_plus_delta_bound) + " --k_max " + str(k_max)
            fname = "batch/submit.batch"
            f = open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname)
            os.system("sbatch " + fname)
