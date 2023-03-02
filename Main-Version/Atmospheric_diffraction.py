import numpy as np
import math
import copy

from astropy.io import fits
from astropy import units as u
from astropy.constants import c
from astropy.units import Quantity

def airmass_to_zenith_dist(airmass):
    """
    Returns zenith distance in degrees: Z = arccos(1/X)
    """

    return np.rad2deg(np.arccos(1. / airmass))


def zenith_dist_to_airmass(zenith_dist):
    """
    ``zenith_dist`` is in degrees
    X = sec(Z)
    """

    return 1./np.cos(np.deg2rad(zenith_dist))


def diff_shift(wave, airmass, atm_ref_wav, conditions):
    """
    Compute the effect of atmospheric difraction at a given airmass
    and reference wavelength

    This function computes the difraction shift at a given wavelength
    relatively to a reference wavelength, using the equations from
    Fillipenko et al (1992). The fonction accepts an array of
    wavelengths for which the difraction shift will be computed.

    Parameters
    ----------
    wave : array
        Input wavelength (in astropy.units)

    atm_ref_wav : float
        Reference wavelength for atmospheric diffraction

    airmass : float
        Airmass of the observation

    conditions: dic
        dictionary of environmmental conditions {Temperature [C],
        Humidity[%], Pressure[mbar]} in astropy.units

    Returns
    -------
    DR : float
        Difraction shift (astropy.units.arcsec)

    Notes
    ----------
    Modified by Myriam Rodrigues (GEPI) from the MOONS ETC (Oscar Gonzalez, ATC UK)
    """

    Lambda0 = atm_ref_wav.to(u.micron).value
    wave = wave.to(u.micron).value

    T = conditions["temperature"].to(u.K, equivalencies=u.temperature()).value
    HR = conditions["humidity"].to(u.dimensionless_unscaled).value
    P = (conditions["pressure"].to(u.mBa)).value

    ZD_deg = airmass_to_zenith_dist(airmass)
    ZD = np.deg2rad(ZD_deg)

    # saturation pressure Ps (millibars)
    PS = -10474.0 + 116.43*T - 0.43284*T**2 + 0.00053840*T**3
    # water vapour pressure
    Pw = HR * PS
    # dry air pressure
    Pa = P - Pw

    #dry air density
    Da = (1 + Pa * (57.90*1.0e-8 - 0.0009325/T + 0.25844/T**2)) * Pa/T

    #1 - P instead of 1 + Pa here? Why? Makes minimal affect of actual values...
    
    #water vapour density ?
    Dw = (1 + Pw * (1 + 3.7 * 1E-4 * Pw) * (- 2.37321 * 1E-3 + 2.23366/T
                                            - 710.792/T**2
                                            + 77514.1/T**3)) * Pw/T
    S0 = 1.0/Lambda0
    S = 1.0/wave

    N0_1 = (1.0E-8*((2371.34+683939.7/(130.0-S0**2)+4547.3/(38.9-S0**2))*Da
            + (6487.31+58.058*S0**2-0.71150*S0**4+0.08851*S0**6)*Dw))

    N_1 = 1.0E-8*((2371.34+683939.7/(130.0-S**2)+4547.3/(38.9-S**2))*Da
                  + (6487.31+58.058*S**2-0.71150*S**4+0.08851*S**6)*Dw)

    
    DR = np.tan(ZD)*(N0_1-N_1) * u.rad
    
    return DR.to(u.arcsec)

def HA_2_ZA(HA,dec):
    lat = np.deg2rad(-24.6272) #Cerro Paranal Latitude
    dec = dec.to(u.rad).value #Declination of target in radians 
    alt=np.arcsin(np.cos(np.deg2rad(np.array(HA)*15))*np.cos(lat)*np.cos(dec)+np.sin(lat)*np.sin(dec)) #altitude of target at HA
    ZA=90-np.rad2deg(alt) #ZA of target at HA
    return ZA


def ZA_2_HA(ZA,dec):
    lat = np.deg2rad(-24.6272) #Cerro Paranal Latitude
    dec = dec.to(u.rad).value #Declination of target in radians   
    alt = np.deg2rad(90 - ZA)
    HA = np.arccos((np.sin(alt)-np.sin(lat)*np.sin(dec))/(np.cos(lat)*np.cos(dec)))
    return np.rad2deg(HA)/15