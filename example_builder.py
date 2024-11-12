#!/usr/bin/env python
# coding: utf-8

import copy
import os
import sys
import time
import warnings

import fsps
import dynesty
import sedpy
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
from astropy.io import fits
from astropy import constants, cosmology, table, units

from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect.observation.observation import Photometry, Spectrum
from prospect.models.templates import TemplateLibrary
from prospect.models import priors, sedmodel

from prospect.fitting import fit_model, lnprobfn

from spectres.spectral_resampling import spectres

import pretty_plot

from proshinbnzvlog import ShockSpecModel, ShockPolySpecModel

cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)

def build_model(
    filename=None , object_redshift=REDSHIFT):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :returns model:
        An instance of prospect.models.SedModel
    """

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["continuity_sfh"]
    
    obs = load_data()
    
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] = object_redshift
    model_params["zred"]["prior"] = priors.Normal(mean=object_redshift, sigma=0.001)

    model_params["logmass"]["prior"] = priors.TopHat(mini=7, maxi=13)

    # --- modify SFH bins ---
    nbins_sfh = 9
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1, 0.0)  # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0),
                                                             scale=np.full(nbins_sfh-1, 0.3),
                                                             df=np.full(nbins_sfh-1, 2))

    # add redshift scaling to agebins, such that t_max = t_univ
    def zred_to_agebins(zred=None, nbins_sfh=None, **extras):
        tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
        ncomp = np.squeeze(nbins_sfh)
        tbinmax = np.squeeze(cosmo.age(20).to("yr").value)
        tbinmax = tuniv - tbinmax
        agelims = [0.0, 7.4772] + np.linspace(8.0, np.log10(tbinmax), ncomp-1).tolist()
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        agebins = zred_to_agebins(zred=zred, **extras)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        nbins = agebins.shape[0]
        sratios = 10**logsfr_ratios
        dt = (10**agebins[:, 1] - 10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses
    
    model_params["logzsol"]["init"] = -0.5
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2., maxi=0.19)

    # --- complexify the dust ---
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.3, sigma=1)
    model_params["dust_index"] = dict(N=1, isfree=True, init=0,
                                      prior=priors.TopHat(mini=-1.0, maxi=0.2))
    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction*dust2
    model_params['dust1'] = dict(N=1, isfree=False, init=0,
                                 prior=None, depends_on=to_dust1)
    model_params['dust1_fraction'] = dict(N=1, isfree=True, init=1.0,
        prior=priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3))

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a statistically acceptable fit.
    model_params['spec_jitter'] = dict(N=1, isfree=True, init=1.0,
                                       prior=priors.TopHat(mini=0.5, maxi=2.0))
    
    # Add dust emission (with fixed dust SED parameters)
    # Since `model_params` is a dictionary of parameter specifications, 
    # and `TemplateLibrary` returns dictionaries of parameter specifications, 
    # we can just update `model_params` with the parameters described in the 
    # pre-packaged `dust_emission` parameter set.
    model_params.update(TemplateLibrary["dust_emission"])
    model_params['duste_gamma']['isfree'] = True
    model_params['duste_qpah']['isfree'] = True
    model_params['duste_umin']['isfree'] = True
    model_params.update(TemplateLibrary["agn"])
    model_params["fagn"]["isfree"] = True
    model_params["agn_tau"]["isfree"] = True
    
    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=0.0, maxi=300.0)
    
    
    # This removes the continuum from the spectroscopy. Highly recommend using when modeling both photometry & spectroscopy
    # order of polynomial that's fit to spectrum
    model_params.update(TemplateLibrary['optimize_speccal'])
    model_params.pop("spec_norm")
    model_params["polyorder"]["init"] = 2

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    """
    model_params['nsigma_outlier_spec'] = dict(N=1, isfree=False, init=50.)
    model_params['f_outlier_spec'] = dict(N=1, isfree=True, init=1e-3,
                                          prior=priors.TopHat(mini=1e-5, maxi=0.01))
    """
    #model_params['nsigma_outlier_phot'] = dict(N=1, isfree=False, init=50.)
    #model_params['f_outlier_phot'] = dict(N=1, isfree=False, init=0.0,
    #                                      prior=priors.TopHat(mini=0, maxi=0.5))

        
        
    # Add nebular emission
    model_params.update(TemplateLibrary["nebular"])
    model_params['gas_logu']['isfree'] = True
    model_params['gas_logu']['init'] = -2.0
    model_params['gas_logz']['isfree'] = True
    model_params['gas_logz']['init'] = -0.3
    _ = model_params["gas_logz"].pop("depends_on")
    model_params['gas_logz']['prior'] = priors.TopHat(mini=-2., maxi=0.19)
    # Adjust for widths of emission lines
    model_params["nebemlineinspec"]["init"] = False
    model_params["eline_sigma"] = {'N': 1, 
                                   'isfree': True, 
                                   'init': 100.0, 'units': 'km/s',
                                   'prior': priors.TopHat(mini=0., maxi=300)}

    model_params['shock_type'] = dict(N=1, isfree=False, init='shock')
    model_params["shock_eline_sigma"] = {
        'N': 1, 'isfree': True, 'init': 100.0, 'units': 'km/s',
        'prior': priors.TopHat(mini=100, maxi=3000)}
    model_params["shock_elum"] = {
        'N': 1, 'isfree': True, 'init': 0.0,
        'prior': priors.TopHat(mini=0., maxi=10)}
    model_params["shock_logB"] = {
        'N': 1, 'isfree': True, 'init': 0.0,
        'prior': priors.TopHat(mini=-4., maxi=1)}
    model_params["shock_logv"] = {
        'N': 1, 'isfree': True, 'init': 2.5,
        'prior': priors.TopHat(mini=2., maxi=3.)}
    model_params["shock_logn"] = {
        'N': 1, 'isfree': True, 'init': 1.0,
        'prior': priors.TopHat(mini=0., maxi=4.)}
    def gas_logz(gas_logz=None, shock_logZ=None, **extras):
        return gas_logz + np.log10(0.0142) - np.log10(0.01542)
    model_params["shock_logZ"] = {
        'N': 1, 'isfree': False, 'init': -0.3, 'depends_on': gas_logz}

    # Now instantiate the model object using this dictionary of parameter specifications
    model = ShockPolySpecModel(model_params)
    #model = ShockSpecModel(model_params)

    return model



def build_sps(zcontinuous=1, compute_vega_mags=False):
    """Load the SPS object.  If add_realism is True, set up to convolve the
    library spectra to an sdss resolution
    """
    from prospect.sources import CSPSpecBasis, FastStepBasis
    sps = FastStepBasis(
        zcontinuous=zcontinuous,
        imf_type=2, imf_lower_limit=0.1, # 1 Chabrier, 2 Kroupa
        compute_vega_mags=compute_vega_mags)
    
    return sps

def build_noise():
    return None, None

def build_all(use_R2700=True, mask_emission=False):

    return (load_data(mask_emission=mask_emission, use_R2700=use_R2700),
        build_model(), build_sps(), build_noise())
