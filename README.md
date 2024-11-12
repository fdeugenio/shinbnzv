# shinbnzv
On-the-fly emission-line fluxes from shock-powered astrophysical gas, with SED interface.


This is described in [F. D'Eugenio et al. (2024; in press).](https://ui.adsabs.harvard.edu/abs/2024arXiv240803982D/abstract)

It calculates emission-line luminosities for a shock-excited gas. Includes the interpolator `shinbnzvlog`, and an SED fitting interface for [`prospector`](https://github.com/bd-j/prospector) v2.0.
The interpolator is provided as static `python3.11` file. Please unpack `static_files.tar.xz`.
There is an example on how to build the prospector model.

Relies on pre-computed [`mappings v`](https://mappings.readthedocs.io/en/latest/#) grids (Sutherland & Dopita, 2017), obtained from the [3MdB](http://3mdb.astro.unam.mx:3686/) (Alarie & Morisset, 2019). The grids are linearly interpolated using the [`qhull`](https://www.math.cmu.edu/users/gleiva/) algorithm (Barber et al. , 1996).The code includes a port to B. D. Johnson's `prospector` (Johnson et al., 2021).

--------------
### Citations.
If you use this software, I would appreciate a citation to [D'Eugenio et al. (2024).](https://ui.adsabs.harvard.edu/abs/2024arXiv240803982D/abstract) Besides, the following citations are mandatory, depending on what is used; if you use the interpolator only, please cite 1., 2., and 5 below. If you also use the [`prospector`](https://github.com/bd-j/prospector) port, please also cite 3. and 4. below. Make sure to follow the guidelines for citing [`prospector`](https://github.com/bd-j/prospector).

### Bibliography
1. Alarie A., Morisset, C., 2019, Rev. Mex. Astron. Astrofis., 55, 377
2. Barber C. B., Dobkin D. P., Huhdanpaa H., 1996, ACM Trans. Math. Softw., 22, 469-483
3. Johnson B. D., Leja J. L., Conroy C., Speagle J. S., 2019, Prospector: Stellar population inference from spectra and SEDs, Astrophysics Source Code Library, record ascl:1905.025
4. Johnson B. D., Leja J., Conroy C., Speagle J. S., 2021, ApJS, 254, 22
5. Sutherland R. S., Dopita M. A., 2017, ApJS, 229, 34
