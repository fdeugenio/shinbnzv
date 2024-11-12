#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""proshinbnzv.py - classes and methods for storing parameters and predicting
observed spectra and photometry from them, given a Source object.
"""

import numpy as np

from sedpy.observate import getSED

from prospect.sources.constants import lightspeed, jansky_cgs

from prospect.models.sedmodel import SpecModel, PolySpecModel

import shinbnzvlog as shock_linterpolator

__all__ = ["ShockSpecModel", "ShockPolySpecModel"]



class ShockSpecModel(SpecModel):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._set_shock_interpolator()

    def _available_parameters(self):
        pars = [("shock_elum",
                 "This float scales the predicted shock nebular emission line"
                 "luminosities, in units of Lsun(Hbeta)/Mformed"),
                ("shock_eline_sigma",
                 "This float gives the velocity dispersion of the shock emission"
                 "lines in km/s"),
                ("shock_type",
                 "This selects the interpolator. Either 'shock' or 'shock+precursor'"),
                ("shock_dust_law",
                 "This selects the shock dust model."),
                ("shock_tau_v",
                 "Shock dust attenuation value"),
                ]

        return pars

    def _set_shock_interpolator(self):
        """Shock line spectrum. 
        normalized to Hbeta.index=59 is Hbeta
        """

        if self.params.get('shock_dust_law', None) is not None:
            self._shock_dust_law = self.params['shock_dust_law'][0]
            self.params['shock_dust_law'] = str(self._shock_dust_law)

        shinds = np.array([
            # Lya, He2, O3, O3,
            # 9, 13, 14, 15,
            # He2, O3, O3,
            13, 14, 15,
            # C3, C3, C2, C2, C2, C2, C2, C2, # Mg2, Mg2
            19, 20, 22, 23, 24, 25, 26, 27, # 31, 32,
            # O2, O2, Ne3, He1, H1, Ne3, H1
            38, 39, 43, 44, 45, 46, 47,
            # S2, S2, Hd, Hg, O3, He14472,
            48, 49, 50, 51, 52, 53,
            # Hb, O3, O3, N1, N1, O15577, N2, He15876,
            59, 61, 62, 63, 64, 67, 68, 69,
            # O1, O1, N2, Ha, N2, He16678,
            70, 72, 73, 74, 75, 76,
            # S2, S2, He17065, Ar37136,
            77, 78, 79, 80,
            # O2, O2, Ar37751, S3, S3
            84, 85, 87, 91, 94,
            # Pag, HeI_10830, Pag, Pab, Paa
            97, 101, 102, 104, 107])
        assert np.abs(self.emline_info["wave"][59] - 4863) < 2, f"Outdated emission-line info; check installation"

        shfluxes = np.ones(len(shinds), dtype=float)

        # Line luminosities all set to 1. This way they must just be scaled
        # by the flux(line)/flux(hb) ratio.
        self._shline_ind = shinds
        self._shline_lum = np.zeros(len(self.emline_info))
        self._shline_lum[shinds] = shfluxes

        shock_type = self.params.get("shock_type", "shock+precursor")

        if shock_type=="shock+precursor":
            self._shock_interpolator = shock_linterpolator.preshock_interp
            return
        if shock_type=="shock":
            self._shock_interpolator = shock_linterpolator.shock_interp
            return
        raise ValueError(f'Parameter {shock_type=} must be "shock" or "shock+precursor"')



    def predict_spec(self, obs, **extras):
        """Generate a prediction for the observed spectrum.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct

          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies
          + ``_eline_wave`` and ``_eline_lum`` - emission line parameters from the SPS model

        It generates the following attributes

          + ``_outwave`` - Wavelength grid (observed frame)
          + ``_speccal`` - Calibration vector
          + ``_sed`` - Intrinsic spectrum (before cilbration vector applied but including emission lines)

        And the following attributes are generated if nebular lines are added

          + ``_fix_eline_spec`` - emission line spectrum for fixed lines, intrinsic units
          + ``_fix_eline_spec`` - emission line spectrum for fitted lines, with
            spectroscopic calibration factor included.

        Numerous quantities related to the emission lines are also cached (see
        ``cache_eline_parameters()`` and ``fit_mle_elines()`` for details.)

        :param obs:
            An observation dictionary, containing the output wavelength array,
            the photometric filter lists, and the observed fluxes and
            uncertainties thereon.  Assumed to be the result of
            :py:meth:`utils.obsutils.rectify_obs`

        :param sigma_spec: (optional)
            The covariance matrix for the spectral noise. It is only used for
            emission line marginalization.

        :returns spec:
            The prediction for the observed frame spectral flux these
            parameters, at the wavelengths specified by ``obs['wavelength']``,
            including multiplication by the calibration vector.
            ndarray of shape ``(nwave,)`` in units of maggies.
        """
        # redshift wavelength
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        self._outwave = obs.get('wavelength', obs_wave)
        if self._outwave is None:
            self._outwave = obs_wave

        # --- cache eline parameters ---
        nsigma = 5 * np.max(self.params.get("shock_eline_sigma", 100.0) / self.params.get("eline_sigma", 100))
        self.cache_eline_parameters(obs, nsigma=nsigma)

        smooth_spec = self.velocity_smoothing(obs_wave, self._norm_spec)
        smooth_spec = obs.instrumental_smoothing(obs_wave, smooth_spec,
                                                 libres=self._library_resolution)

        # --- add fixed lines ---
        assert self.params["nebemlineinspec"] == False, "Must add shock and nebular lines within prospector"
        assert self.params.get("marginalize_elines", False) == False, "Cannot marginalise lines when shock lines included"
        emask = self._fix_eline_pixelmask
        if emask.any():
            # Add SF lines
            inds = self._fix_eline & self._valid_eline
            espec = self.predict_eline_spec(line_indices=inds,
                                            wave=self._outwave[emask])
            self._fix_eline_spec = espec
            smooth_spec[emask] += self._fix_eline_spec.sum(axis=1)

            # Add shock lines
            shspec = self.predict_shline_spec(line_indices=inds,
                                              wave=self._outwave[emask], obs=obs)
            self._shock_eline_spec = shspec
            smooth_spec[emask] += self._shock_eline_spec

        # --- calibration ---
        self._speccal = self.spec_calibration(obs=obs, spec=smooth_spec, **extras)
        calibrated_spec = smooth_spec * self._speccal

        # --- cache intrinsic spectrum ---
        self._sed = calibrated_spec / self._speccal

        return calibrated_spec

    def predict_lines(self, obs, **extras):
        """Generate a prediction for the observed nebular line fluxes, including
        Shocks.

        :param obs:
            A ``data.observation.Lines()`` instance, with the attributes
            + ``"wavelength"`` - the observed frame wavelength of the lines.
            + ``"line_ind"`` - a set of indices identifying the observed lines in
            the fsps line array

        :returns elum:
            The prediction for the observed frame nebular + AGN emission line
            flux these parameters, at the wavelengths specified by
            ``obs['wavelength']``, ndarray of shape ``(nwave,)`` in units of
            erg/s/cm^2.
        """
        sflums = super().predict_lines(obs, **extras)

        logB = self.params.get('shock_logB',  0. )
        logv = self.params.get('shock_logv',  2.5)
        logn = self.params.get('shock_logn',  1. )
        logZ = self.params.get('shock_logZ', -0.3) + np.log10(0.01524)

        shnorm = self.params.get('shock_elum', 1.0) * self.flux_norm() / (1 + self._zred)
        shlums = self._shline_lum[line_indices] * shnorm

        flux_rats = 10**self._shock_interpolator(logB, logn, logZ, logv)
        shlums[self._shline_ind] *= flux_rats

        elums = sflums + shlums

        return elums

    def predict_phot(self, filters):
        """Generate a prediction for the observed photometry.  This method assumes
        that the parameters have been set and that the following attributes are
        present and correct:
          + ``_wave`` - The SPS restframe wavelength array
          + ``_zred`` - Redshift
          + ``_norm_spec`` - Observed frame spectral fluxes, in units of maggies.
          + ``_ewave_obs`` and ``_eline_lum`` - emission line parameters from
            the SPS model

        :param filters:
            Instance of :py:class:`sedpy.observate.FilterSet` or list of
            :py:class:`sedpy.observate.Filter` objects. If there is no
            photometry, ``None`` should be supplied.

        :returns phot:
            Observed frame photometry of the model SED through the given filters.
            ndarray of shape ``(len(filters),)``, in units of maggies.
            If ``filters`` is None, this returns 0.0
        """
        if filters is None:
            return 0.0

        # generate photometry w/o emission lines
        obs_wave = self.observed_wave(self._wave, do_wavecal=False)
        flambda = self._norm_spec * lightspeed / obs_wave**2 * (3631*jansky_cgs)
        phot = 10**(-0.4 * np.atleast_1d(getSED(obs_wave, flambda, filters)))
        # TODO: below is faster for sedpy > 0.2.0
        #phot = np.atleast_1d(getSED(obs_wave, flambda, filters, linear_flux=True))

        # generate emission-line photometry
        if (self._want_lines & self._need_lines):
            phot += self.nebline_photometry(filters)
            # Add shock lines to photometry
            # this could use _use_line
            logB = self.params.get('shock_logB',  0. )
            logv = self.params.get('shock_logv',  2.5)
            logn = self.params.get('shock_logn',  1. )
            logZ = self.params.get('shock_logZ', -0.3) + np.log10(0.01524)
     
            shnorm = self.params.get('shock_elum', 1.0) * self.flux_norm() / (1 + self._zred) * (3631*jansky_cgs)
            shlums = self._shline_lum * shnorm
     
            flux_rats = 10**self._shock_interpolator(logB, logn, logZ, logv)[0]
            shlums[self._shline_ind] *= flux_rats

            shlams = self._ewave_obs
            if (tau_v:=self.params.get("shock_tau_v", None)) is not None:
                a_v = 2.5*np.log10(np.e) * tau_v
                shlums[self._shline_ind] *= self._shock_dust_law(
                    shlams[self._shline_ind]/(1+self._zred), a_v)
     
            phot += self.nebline_photometry(filters, shlams, shlums)

        return phot

    def predict_shline_spec(self, line_indices, wave, obs):
        # HACK to change the AGN line widths.
        nsigma = 5 * np.max(self.params.get("shock_eline_sigma", 100.0) / self.params.get("eline_sigma", 100))
        _orig_eline_sigma_kms_ = self._eline_sigma_kms
        self.params['eline_sigma'] = self.params.get('shock_eline_sigma', 100.)
        self.cache_eline_parameters(obs, nsigma=nsigma)

        logB = self.params.get('shock_logB',  0. )
        logv = self.params.get('shock_logv',  2.5)
        logn = self.params.get('shock_logn',  1. )
        logZ = self.params.get('shock_logZ', -0.3) + np.log10(0.01524)
        
        flux_rats = 10**self._shock_interpolator(logB, logn, logZ, logv)[0]

        # HACK to change the AGN line widths.
        nline = self._ewave_obs.shape[0]
        gaussians = self.get_eline_gaussians(lineidx=line_indices, wave=wave)

        shnorm = self.params.get('shock_elum', 1.0) * self.flux_norm() / (1 + self._zred)

        shlums = np.copy(self._shline_lum)
        shlums[self._shline_ind] *= flux_rats
        shlums = shlums[line_indices] * shnorm
        shline_spec = (shlums * gaussians).sum(axis=1)

        self.params['eline_sigma'] = _orig_eline_sigma_kms_
        self._eline_sigma_kms = _orig_eline_sigma_kms_

        # --- smooth and put on output wavelength grid ---
        if (tau_v:=self.params.get("shock_tau_v", None)) is not None:
            a_v = 2.5*np.log10(np.e) * tau_v
            shline_spec *= self._shock_dust_law(wave/(1+self._zred), a_v)

        return shline_spec



    def old_predict_shline_spec(self, line_indices, wave):
        # HACK to change the AGN line widths.
        logB = self.params.get('shock_logB',  0. )
        logv = self.params.get('shock_logv',  2.5)
        logn = self.params.get('shock_logn',  1. )
        logZ = self.params.get('shock_logZ', -0.3) + np.log10(0.01524)
        
        flux_rats = 10**self._shock_interpolator(logB, logn, logZ, logv)[0]

        # HACK to change the AGN line widths.
        orig = self._eline_sigma_kms
        nline = self._ewave_obs.shape[0]
        self._eline_sigma_kms = np.atleast_1d(self.params.get('shock_eline_sigma', 100.0))
        self._eline_sigma_kms = (self._eline_sigma_kms[None] * np.ones(nline)).squeeze()
        #self._eline_sigma_kms *= np.ones(self._ewave_obs.shape[0])
        gaussians = self.get_eline_gaussians(lineidx=line_indices, wave=wave)
        self._eline_sigma_kms = orig

        shnorm = self.params.get('shock_elum', 1.0) * self.flux_norm() / (1 + self._zred)

        shlums = np.copy(self._shline_lum)
        shlums[self._shline_ind] *= flux_rats
        shlums = shlums[line_indices] * shnorm
        shline_spec = (shlums * gaussians).sum(axis=1)

        return shline_spec



class ShockPolySpecModel(PolySpecModel, ShockSpecModel):
    pass



def ln_mvn(x, mean=None, cov=None):
    """Calculates the natural logarithm of the multivariate normal PDF
    evaluated at `x`

    :param x:
        locations where samples are desired.

    :param mean:
        Center(s) of the gaussians.

    :param cov:
        Covariances of the gaussians.
    """
    ndim = mean.shape[-1]
    dev = x - mean
    log_2pi = np.log(2 * np.pi)
    sign, log_det = np.linalg.slogdet(cov)
    exp = np.dot(dev.T, np.dot(np.linalg.pinv(cov, rcond=1e-12), dev))

    return -0.5 * (ndim * log_2pi + log_det + exp)


def gauss(x, mu, A, sigma):
    """Sample multiple gaussians at positions x.

    :param x:
        locations where samples are desired.

    :param mu:
        Center(s) of the gaussians.

    :param A:
        Amplitude(s) of the gaussians, defined in terms of total area.

    :param sigma:
        Dispersion(s) of the gaussians, un units of x.

    :returns val:
        The values of the sum of gaussians at x.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = A / (sigma * np.sqrt(np.pi * 2)) * np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2))
    return val.sum(axis=-1)
