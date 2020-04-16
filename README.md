# Dark Photon Oscillations in Our Inhomogeneous Universe

Code repository for the paper
**Dark Photon Oscillations in Our Inhomogeneous Universe**
by Andrea Caputo, Hongwan Liu, Siddharth Mishra-Sharma, and Joshua T. Ruderman.

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Dark Photons](https://img.shields.io/badge/Photons-Dark-yellowgreen.svg)](./)
[![Dark Matter](https://img.shields.io/badge/Matter-Dark-black.svg)](./)
[![arXiv](https://img.shields.io/badge/arXiv-2002.05165%20-green.svg)](https://arxiv.org/abs/2002.05165)
[![arXiv](https://img.shields.io/badge/arXiv-2004.06733%20-green.svg)](https://arxiv.org/abs/2002.xxxxx)

![Illustration of photon/dark photon passage through inhomogeneities.](notebooks/animations/simulation_animation.gif)


## Abstract

A dark photon may kinetically mix with the ordinary photon, inducing oscillations with observable imprints on cosmology.  Oscillations are resonantly enhanced if the dark photon mass equals the ordinary photon plasma mass, which tracks the free electron number density.  Previous studies have assumed a homogeneous Universe; in this Letter, we introduce for the first time an analytic formalism for treating resonant oscillations in the presence of inhomogeneities of the photon plasma mass.  We apply our formalism to determine constraints from Cosmic Microwave Background photons oscillating into dark photons, and from heating of the primordial plasma due to dark photon dark matter converting into low-energy photons. Including the effect of inhomogeneities demonstrates that prior homogeneous constraints are not conservative, and simultaneously extends current experimental limits into a vast new parameter space.

## Main Results

The data points for the fiducial constraints presented in the paper are provided in [data/constraints](data/constraints), and a Jupyter notebook plotting these is provided in [10_fiducial_constraints.ipynb](notebooks/10_fiducial_constraints.ipynb).

![Constraints on dark photons and dark photon dark matter.](paper/draft-letter/plots/results_web.png)

## Code

The dependencies of the code are listed in [environments.yml](environment.yml).

The [notebooks](notebooks/) folder contains various Jupyter notebooks that reproduce the plots in the paper. Additionally, [00_dP_dz.ipynb](notebooks/00_dP_dz.ipynb) describes how to extract the differential conversion probability dP/dz for various scenarios and choices of the underlying matter distribution explored in the papers.

## Authors

-  Andrea Caputo; andrea dot caputo at uv dot es
-  Hongwan Liu; hongwanl at princeton dot edu
-  Siddharth Mishra-Sharma; sm8383 at nyu dot edu
-  Joshua T. Ruderman; ruderman at nyu dot edu

## Citation

If you use this code, please cite our papers:

Constraints on dark photons and dark photon dark matter:

```
@article{Caputo:2020bdy,
    author = "Caputo, Andrea and Liu, Hongwan and Mishra-Sharma, Siddharth and Ruderman, Joshua T.",
    archivePrefix = "arXiv",
    eprint = "2002.05165",
    month = "2",
    primaryClass = "astro-ph.CO",
    title = "{Dark Photon Oscillations in Our Inhomogeneous Universe}",
    year = "2020"
}
```

Formalism and methodology:

```
@article{Caputo:2020rnx,
    author = "Caputo, Andrea and Liu, Hongwan and Mishra-Sharma, Siddharth and Ruderman, Joshua T.",
    archivePrefix = "arXiv",
    eprint = "2004.06733",
    month = "4",
    primaryClass = "astro-ph.CO",
    title = "{Modeling Dark Photon Oscillations in Our Inhomogeneous Universe}",
    year = "2020"
}
```
