# Bispectra from hydrodynamical simulations

This repository contains power spectrum and bispectrum measurements presented in the following paper:

> Foreman, Coulton, Villaescusa-Navarro, and Barreira, *Baryonic effects on the matter bispectrum*, 2019, arXiv:[1910.03597](https://arxiv.org/abs/1910.03597)

These measurements were made with the [bskit](github.com/sjforeman/bskit) code from the IllustrisTNG, Illustris, EAGLE, and BAHAMAS simulation suites at redshifts 0, 1, 2, and 3. Please refer to the paper for more details about the measurement procedures and choices involved.

Note that a wider variety of simulation power spectra (including from the BAHAMAS runs we use here, but up to higher wavenumbers and at more redshifts) can be found at Marcel van Daalen's website: [http://powerlib.strw.leidenuniv.nl](http://powerlib.strw.leidenuniv.nl).

## File formats

### Power spectra (`ps/`)

Auto spectra file names have the following format:

`SIMULATION_FIELD_NGRID_zZ_ps_DK_KMIN.dat`,

where `SIMULATION` is the simulation name (`_DM` or `-Dark` refers to a dark matter only [DMO] simulation, while the others are from the full hydro runs), `FIELD` is the overdensity field (total matter, dark matter, or baryons), `NGRID` is the resolution of the Fourier grid used for the measurement, `Z` is redshift, `DK` is the *k* bin size (where `kf` refers to the fundamental mode *k_f = 2pi/Lbox*), and `KMIN` is the lower edge of the lowest *k* bin. Cross spectra are named analogously. 

Each file has the following columns:

1. mean *k* over bin [h Mpc^-1]
2. mean *P* over bin [h^-3 Mpc^3]
3. number of modes in bin [dimensionless]

### Power spectrum ratio errorbars (`ps_ratio_errs/`)

These files contain estimates of the sample variance uncertainty in ratios of the measured power spectra (hydro/DMO), determined from subboxes of each simulation using the method described in the paper above. They are named analogously to the corresponding power spectrum files, and have the following columns:

1. mean *k* over subbox bin [h Mpc^-1]
2. uncertainty in P_hydro/P_DMO in this bin [dimensionless]

Note that the *k*-bin edges used to estimate these uncertainties are identical to those used for the full-box auto spectrum measurements, such that there is a one-to-one correspondence between auto spectrum and errorbar files if the first auto spectrum bin is ignored. For instance, comparing the first few lines of the TNG300 total-matter power spectrum at *z=0*,

```
# k_mean_over_bin [h Mpc^-1] P_bin [h^-3 Mpc^3] N_modes
3.911336014668146538e-02 1.219744010416666606e+04 1.800000000000000000e+01
6.837341045179674837e-02 1.037742036290322540e+04 6.200000000000000000e+01
9.606098566128283556e-02 4.468139987244897384e+03 9.800000000000000000e+01
1.244554894311087484e-01 4.256158333333333758e+03 2.100000000000000000e+02
1.562393171446664064e-01 3.031594285714285888e+03 3.500000000000000000e+02
```

and the corresponding errorbar file,

```
# k_mean_over_subbox_bin [h Mpc^-1] sig_ratio [dimensionless]
6.129937e-02 2.722261e-04
9.448369e-02 1.503153e-04
1.341754e-01 1.561539e-04
1.501522e-01 1.821392e-04
```

lines 3,4,5,6,… of the power spectrum file correspond to lines 2,3,4,5,… of the errorbar file. (The mean *k* values computed from the subbox differ slightly from those in the full box due to the different discrete sets of modes contained in each box.)


### Bispectra (`bs/`)

File names have a similar format to power spectra:

`SIMULATION_FIELD_NGRID_zZ_bs_DKLOW_DKHIGH_NUMlowbins.dat`,

where each triangle side is binned in `NUM` bins of width `DKLOW`, and in bins of width `DKHIGH` at higher *k*.

Each line of a file corresponds to a bin of triangles with side lengths *k_1,k_2,k_3* where each *k_i* is within a specified wavenumber bin. The columns are as follows:

1. triangle index [dimensionless]
2. mean *k_1* value over triangle bin [h Mpc^-1]
3. mean *k_2* value over triangle bin [h Mpc^-1]
4. mean *k_3* value over triangle bin [h Mpc^-1]
5. lower *k_1* bin edge [h Mpc^-1]
6. upper *k_1* bin edge [h Mpc^-1]
7. lower *k_2* bin edge [h Mpc^-1]
8. upper *k_2* bin edge [h Mpc^-1]
9. lower *k_3* bin edge [h Mpc^-1]
10. upper *k_3* bin edge [h Mpc^-1]
11. mean *B* over bin [h^6 Mpc^-6]
12. number of modes in bin [dimensionless]

Each triangle index corresponds to a unique set of (*k_1,k_2,k_3*) bins.


### Bispectrum ratio errorbars (`bs_ratio_errs/`)

As with the power spectrum these files contain estimates of the sample variance uncertainty in ratios of the measured bispectra (hydro/DMO). Their columns are as follows:

1. triangle index [dimensionless]
2. uncertainty in B_hydro/B_DMO in this bin [dimensionless]

Triangle indices can be used to match these values to the corresponding triangle bins and bispectrum values in the bispectrum files. Bins for which we were unable to quantify sample-variance uncertainties from subboxes have `nan` values.

## References

If any of these measurements are used in original research, please cite the associated paper:

> Foreman, Coulton, Villaescusa-Navarro, and Barreira, *Baryonic effects on the matter bispectrum*, 2019, arXiv:[1910.03597](https://arxiv.org/abs/1910.03597)

In addition, to recognize the tremendous amount of work that has gone into each simulation suite by the respective team, please cite the following papers (at minimum) if measurements from the corresponding simulations are used:

### IllustrisTNG (http://www.tng-project.org)

> Pillepich et al., *First results from the IllustrisTNG simulations: the stellar mass content of groups and clusters of galaxies*, [Mon. Not. Roy. Astron. Soc., 475, 648 (2018)](https://dx.doi.org/10.1093/mnras/stx3112), arXiv:[1707.03406](https://arxiv.org/abs/1707.03406)

> Springel et al., *First results from the IllustrisTNG simulations: matter and galaxy clustering*, [Mon. Not. Roy. Astron. Soc., 475, 676 (2018)](https://dx.doi.org/10.1093/mnras/stx3304), arXiv:[1707.03397](https://arxiv.org/abs/1707.03397)

> Nelson et al., *First results from the IllustrisTNG simulations: the galaxy color bimodality*, [Mon. Not. Roy. Astron. Soc., 475, 624 (2018)](https://dx.doi.org/10.1093/mnras/stx3040), arXiv:[1707.03395](https://arxiv.org/abs/1707.03395)

> Naiman et al., *First results from the IllustrisTNG simulations: A tale of two elements -- chemical evolution of magnesium and europium*, [Mon. Not. Roy. Astron. Soc., 475, 624 (2018)](https://dx.doi.org/10.1093/mnras/sty618), arXiv:[1707.03401](https://arxiv.org/abs/1707.03401)

> Marinacci et al., *First results from the IllustrisTNG simulations: radio haloes and magnetic fields*, [Mon. Not. Roy. Astron. Soc., 475, 624 (2018)](https://dx.doi.org/10.1093/mnras/sty2206), arXiv:[1707.03396](https://arxiv.org/abs/1707.03396)

> Nelson et al., *The IllustrisTNG Simulations: Public Data Release*, 2018, arXiv:[1812.05609](https://arxiv.org/abs/1812.05609)

### Illustris (http://www.illustris-project.org)

> Vogelsberger et al.,  *A model for cosmological simulations of galaxy formation physics*, [Mon. Not. Roy. Astron. Soc., 436, 3031 (2013)](https://dx.doi.org/10.1093/mnras/stt1789), arXiv:[1305.2913](https://arxiv.org/abs/1305.2913) 

> Vogelsberger  et al.,  *Properties of galaxies reproduced by a hydrodynamic simulation*,  [Nature, 509, 177 (2014)](https://dx.doi.org/10.1038/nature13316), arXiv:[1405.1418](https://arxiv.org/abs/1405.1418) 

> Genel et al., *Introducing the Illustris Project: the evolution of galaxy populations across cosmic time*, [Mon. Not. Roy. Astron. Soc., 445, 175 (2014)](https://dx.doi.org/10.1093/mnras/stu1654), arXiv:[1405.3749](https://arxiv.org/abs/1405.3749)

> Sijacki et al., *The Illustris simulation: the evolving population of black holes across cosmic time*,  [Mon. Not. Roy. Astron. Soc., 452, 575 (2015)](https://dx.doi.org/10.1093/mnras/stv1340), arXiv:[1408.6842](https://arxiv.org/abs/1408.6842) 

### EAGLE (http://icc.dur.ac.uk/Eagle/)

> Schaye et al., *The EAGLE project: Simulating the evolution and assembly of galaxies and their environments*, [Mon. Not. Roy. Astron. Soc., 446, 521 (2015)](https://dx.doi.org/10.1093/mnras/stu2058), arXiv:[1407.7040](https://arxiv.org/abs/1407.7040) 

> Crain et al., *The EAGLE simulations of galaxy formation: calibration of subgrid physics and model variations*, [Mon. Not. Roy. Astron. Soc., 450, 1937 (2015)](https://doi.org/10.1093/mnras/stv725), arXiv:[1501.01311](https://arxiv.org/abs/1501.01311) 

> McAlpine et al., *The EAGLE simulations of galaxy formation: public release of halo and galaxy catalogues*, [Astron. Comput., 15, 72 (2016)](https://doi.org/10.1016/j.ascom.2016.02.004), arXiv:[1510.01320](https://arxiv.org/abs/1510.01320) 

> The EAGLE team, *The EAGLE simulations of galaxy formation: Public release of particle data*, 2017, arXiv:[1706.09899](https://arxiv.org/abs/1706.09899) 

### BAHAMAS (http://www.astro.ljmu.ac.uk/~igm/BAHAMAS/)

> McCarthy et al., *The BAHAMAS project: Calibrated hydrodynamical simulations for large-scale structure cosmology*, [Mon. Not. Roy. Astron. Soc., 465, 2936 (2017)](https://dx.doi.org/10.1093/mnras/stw2792), arXiv:[1603.02702](https://arxiv.org/abs/1603.02702) 

> McCarthy et al., *The BAHAMAS project: the CMB--large-scale structure tension and the roles of massive neutrinos and galaxy formation*, [Mon. Not. Roy. Astron. Soc., 476, 2999 (2018)](https://dx.doi.org/10.1093/mnras/sty377), arXiv:[1712.02411](https://arxiv.org/abs/1712.02411) 
