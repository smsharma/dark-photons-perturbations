
import sys, os
import random
import numpy as np

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu


cd /home/sm8383/spectral_distortions_perturbations/cluster

'''

iz_compute_ary = np.arange(500)

for iz_compute in iz_compute_ary:
    batchn = batch  + "\n"
    batchn += "python gen_pk.py --iz_compute " + str(iz_compute)
    fname = "batch/z_compute_" + str(iz_compute) + ".batch" 
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname)
    os.system("sbatch " + fname)
