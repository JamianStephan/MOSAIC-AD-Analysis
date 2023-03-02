from scipy.integrate import quad,dblquad
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import copy, math
from astropy import units as u
from astropy.constants import c
from astropy.units import Quantity
import matplotlib as mpl

def atmosdisp(wave, wave_0, za, pressure, temp, water=2., fco2=0.0004, obsalt=0.):
    """:NAME:
         atmosdisp
     PURPOSE:
         Compute the atmosperic dispersion relative to lambda_0.     
     CATEGORY:
         Spectroscopy
     CALLING SEQUENCE:
         result = atmosdisp(wave,wave_0,za,pressure,temp,[water],[obsalt],$
                            CANCEL=cancel)
     INPUTS:
         wave     - wavelength in microns
         wave_0   - reference wavelength in microns
         za       - zenith angle of object [in degrees]
         pressure - atmospheric pressure in mm of Hg
         temp     - atmospheric temperature in degrees C
     OPTIONAL INPUTS:
         water    - water vapor pressure in mm of Hg.
         fco2     - relative concentration of CO2 (by pressure)
         obsalt    - The observatory altitude in km.
     KEYWORD PARAMETERS:
         CANCEL - Set on return if there is a problem
     OUTPUTS:
         Returns the atmospheric disperion in arcseconds.      
     PROCEDURE:
         Computes the difference between the dispersion at two
         wavelengths.  The dispersion for each wavelength is derived from
         Section 4.3 of Green's "Spherical Astronomy" (1985).
     EXAMPLE:
     MODIFICATION HISTORY:
         2000-04-05 - written by M. Cushing, Institute for Astronomy, UH
         2002-07-26 - cleaned up a bit.
         2003-10-20 - modified formula - WDV
         2011-10-07 15:51 IJMC: Converted to Python, with some unit conversions
    -"""

    #function atmosdisp,wave,wave_0,za,pressure,temp,water,obsalt,CANCEL=cancel

    # Constants
    
    
    mmHg2pa = 101325./760.      # Pascals per Torr (i.e., per mm Hg)
    rearth = 6378.136e6 #6371.03	# mean radius of earth in km [Allen's]
    hconst = 2.926554e-2	# R/(mu*g) in km/deg K,  R=gas const=8.3143e7
                                    # mu=mean mol wght. of atm=28.970, g=980.665
    tempk  = temp + 273.15
    pressure_pa = pressure * mmHg2pa
    water_pp = water/pressure   # Partial pressure
    hratio = (hconst * tempk)/(rearth + obsalt)

    # Compute index of refraction

    nindx  = nAir(wave,P=pressure_pa,T=tempk,pph2o=water_pp, fco2=fco2)
    nindx0 = nAir(wave_0,P=pressure_pa,T=tempk,pph2o=water_pp, fco2=fco2)

    # Compute dispersion

    acoef  = (1. - hratio)*(nindx - nindx0)
    bcoef  = 0.5*(nindx*nindx - nindx0*nindx0) - (1. + hratio)*(nindx - nindx0)

    tanz   = np.tan(np.deg2rad(za))
    disp   = 206265.*tanz*(acoef + bcoef*tanz*tanz)

    #print nindx
    #print nindx0
    #print acoef
    #print bcoef
    #print tanz
    #print disp
    return disp

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



def nAir(vaclam, T=293.15, P=1e5, fco2=0.0004, pph2o=0.):
    """Return the index of refraction of air at a given wavelength.

    :INPUTS: 

       vaclam: scalar or Numpy array
              Vacuum wavelength (in microns) at which to calculate n
    
       T : scalar
           temperature in Kelvin
       
       P : scalar
           pressure in Pascals

       fc02 : scalar
           carbon dioxide content, as a fraction of the total atmosphere

       pph2o : scalar
           water vapor partial pressure, in Pascals

    :REFERENCE: 
specfi        Boensch and Potulski, 1998 Metrologia 35 133
    """
    # 2011-10-07 15:14 IJMC: Created
    # 2012-12-05 20:47 IJMC: Explicitly added check for 'None' option inputs.

    if T is None:
        T = 293.15
    if P is None:
        P = 1e5
    if fco2 is None:
        fco2 = 0.0004
    if pph2o is None:
        pph2o = 0.0

    sigma = 1./vaclam
    sigma2 = sigma * sigma

    # (Eq. 6a)
    nm1_drystp =  1e-8 * (8091.37 + 2333983. / (130. - sigma2) + 15518. / (38.9 - sigma2))

    # Effect of CO2 (Eq. 7):
    nm1_dryco2 = nm1_drystp * (1. + 0.5327 * (fco2 - 0.0004))

    # Effect of temperature and pressure (Eq. 8):
    nm1_dry = ((nm1_dryco2 * P) * 1.0727933e-5) * \
        (1. + 1e-8 * (-2.10233 - 0.009876 * T) * P) / (0.0036610 * T)

    # Effect of H2O (Eq. 9):
    try:
        n = 1. + (nm1_dry - pph2o * (3.8020 - 0.0384 * sigma2) * 1e-10).astype(float64)
    except:
        n = 1. + (nm1_dry - pph2o * (3.8020 - 0.0384 * sigma2) * 1e-10)

    return n

def overlap (fract_sep):
    import numpy as np
    a=fract_sep/2.
#    if (a >= 1.0):
#        return 0.

    b=2*np.arccos(a)/np.pi
    c=2*a*np.sqrt(1-a*a)/np.pi

    d=b-c
    d[np.where(a>=2.0)] = 0.

    return d

def band_throughput(lc,ll,lh,zd):
    interval = (lh-ll)/99.
    lr = np.arange(ll,lh,interval)
    disp = atmosdisp(lr,lc,zd,534.04,11.5,obsalt=3.,water=2.)

def moffat_xy(y,x,alpha,betam,offset):
    r=sqrt((x-offset)**2 + y*y)
    if alpha >=0. and betam>1.:
        norm = (2*pi*alpha**2)/(2*(betam-1))
    else:
        raise ValueError('alpha and/or beta out of bounds')
    return pow(1.+pow(r/alpha,2),-betam)/norm

def gaussian_xy(y,x,s,offset):
    r=sqrt((x-offset)**2+y*y)
    return exp(-pow(r,2)/(2*s*s))/2*pi*s*s

def lightfrac(seeing,offset,fibreD,profile='Gaussian',beta=2.):
    rfib = fibreD/2.
    if (profile == 'Gaussian'):
        s = seeing/2.3548
        lf=dblquad(gaussian_xy, -rfib,rfib,lambda x: - sqrt(rfib*rfib-x*x), lambda x: sqrt(rfib*rfib-x*x),args=(s,offset))[0]
    elif profile=='Moffat':
        alpha = seeing/(2.*sqrt(2**(1./beta)-1.))
        lf=dblquad(moffat_xy,-rfib,rfib,lambda x: -sqrt(rfib*rfib-x*x), lambda x: sqrt(rfib*rfib-x*x), args=(alpha,beta,offset))[0]
    return lf

def lightfrac_old(seeing,fiberD,offset):
    # note that the integral of a circularly-symmetric gaussian over 0 
    # to 2pi and 0 to infinity is just 2 pi sigma^2...
    s=seeing/2.3548
    rfib=fiberD/2.
    lf=quad(lambda x: x*exp(-pow(x-offset,2)/(2*s*s))/ \
            (s*(s+offset*sqrt(pi/2.))), 0, rfib)[0]
    return lf

def wavethrough(wcen,wmeas,zd,p=534.04,t=11.5,alt=3.,water=2,seeing=0.5,apert=0.7,beta=2.5):
    zz=atmosdisp(wmeas,wcen,zd,p,t,obsalt=alt,water=water)
    return lightfrac(seeing,zz,apert,profile='Moffat',beta=beta)

def plotall(wl,zdrange,wcen,apert=0.7,seeing=0.5,p=534.04,t=11.5,title="title"):
    fig, ax = plt.subplots(figsize=(10,7))
    plt.axvline(wcen,color='black',linewidth=0.5,label='Fibre Centre Wavelength')
    weights = np.arange(1, len(zdrange)+1)
    norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
    print(t)
    print(p)
    k=0
    for i in zdrange:
        tp = []
        for j in wl:
            tp.append(wavethrough(wcen,j,i,apert=apert,seeing=seeing,p=p,t=t))
        tt = asarray(tp)
        if (k == 0):
            t0 = tt
        k = k + 1
        plt.plot(wl,tt/t0,label=str(i),color=cmap.to_rgba(k))
    plt.legend()
    plt.title(title)
    plt.ylim(0,1.1)
    return

def plotadc(globoff,zd):
    tp=[]
    wl = asarray([0.77,0.85,1.01,1.045,1.37,1.42,1.67,1.926])
    off = asarray([-0.25,-0.05,0.19,0.21,0.25,0.24,0.08,-0.21])
    off = off/3.15 
    t0 = lightfrac(0.5,0.,0.6)
    zdr = asarray([0,30,45, 50,60])
    scale = asarray([0.,186,316,374,540])
    for i in range(0,5,1):
        if (zd == zdr[i]):
            off = off*scale[i]/540.
            globoff = globoff*scale[i]/540
    for i in range(0,8,1):
        tp.append(lightfrac(0.5,off[i]+globoff,0.6))
    tt=asarray(tp)
    print(wl)
    print(off)
    print(tt)
    plt.plot(wl,tt/t0,label=str(zd))
    return
                

def allplots():
    zd=arange(0,65,5)
    wl =arange(0.77,1.8,0.02)
    plotall(wl,zd,1.0,apert=0.6)
    plt.legend()
    plt.title('NIR All')
    plt.ylim(0,1.2)
    plt.savefig('NIR_All.png')
    plt.close()
    wl = arange(0.38,0.50,0.01)
    plotall(wl,zd,0.44)
    plt.legend()
    plt.title('Vis LR 1')
    plt.ylim(0,1.2)
    plt.savefig('Vis_LR_1.png')
    plt.close()
    wl = arange(0.487,0.637,0.01)
    plotall(wl,zd,0.545)
    plt.legend()
    plt.title('Vis LR 2')
    plt.ylim(0,1.2)
    plt.savefig('Vis_LR_2.png')
    plt.close()
    wl = arange(0.627,0.821,0.01)
    plotall(wl,zd,0.71)
    plt.legend()
    plt.title('Vis LR 3')
    plt.ylim(0,1.2)
    plt.savefig('Vis_LR_3.png')
    plt.close()
    wl = arange(0.38,0.821,0.05)
    plotall(wl,zd,0.545)
    plt.legend()
    plt.title('Vis All')
    plt.ylim(0,1.2)
    plt.savefig('Vis_All.png')
    plt.close()
    wl = arange(0.77,1.045,0.03)
    plotall(wl,zd,0.88,apert=0.6)
    plt.legend()
    plt.title('NIR LR 1')
    plt.ylim(0,1.2)
    plt.savefig('NIR_LR_1.png')
    plt.close()
    wl = arange(1.01,1.37,0.03)
    plotall(wl,zd,1.16,apert=0.6)
    plt.legend()
    plt.title('NIR LR 2')
    plt.ylim(0,1.2)
    plt.savefig('NIR_LR_2.png')
    plt.close()
    wl = arange(1.42,1.926,0.03)
    plotall(wl,zd,1.65,apert=0.6)
    plt.legend()
    plt.title('NIR LR 3')
    plt.ylim(0,1.2)
    plt.savefig('NIR_LR_3.png')
    plt.close()
    
    
def adcplot():
    import numpy as np
    import matplotlib.pyplot as plt
    zd=np.arange(0,60,1)
    lr1=atmosdisp(0.38,0.497,zd,534.04,11.5,obsalt=3.,water=2.)
    lr2=atmosdisp(0.487,0.637,zd,534.04,11.5,obsalt=3.,water=2.)
    lr3=atmosdisp(0.627,0.821,zd,534.04,11.5,obsalt=3.,water=2.)
    lr4=atmosdisp(0.38,0.821,zd,534.04,11.5,obsalt=3.,water=2.)
    lr5=atmosdisp(0.77,1.063,zd,534.04,11.5,obsalt=3.,water=2.)
    lr6=atmosdisp(1.01,1.395,zd,534.04,11.5,obsalt=3.,water=2.)
    lr7=atmosdisp(1.42,1.857,zd,534.04,11.5,obsalt=3.,water=2.)
    lr8=atmosdisp(0.77,1.857,zd,534.04,11.5,obsalt=3.,water=2.)

    hr1=atmosdisp(0.770,0.907,zd,534.04,11.5,obsalt=3.,water=2.)
    hr2=atmosdisp(1.523,1.658,zd,534.04,11.5,obsalt=3.,water=2.)
    hr3=atmosdisp(0.77,1.658,zd,534.04,11.5,obsalt=3.,water=2.)
    hr4=atmosdisp(0.38,0.821,zd,534.04,11.5,obsalt=3.,water=2.)
    hr5=atmosdisp(0.77,1.063,zd,534.04,11.5,obsalt=3.,water=2.)
    hr6=atmosdisp(1.01,1.395,zd,534.04,11.5,obsalt=3.,water=2.)
    
    weights4 = np.arange(0, 4+1)
    norm4 = mpl.colors.Normalize(vmin=min(weights4), vmax=max(weights4))
    cmap4_1 = mpl.cm.ScalarMappable(norm=norm4, cmap=mpl.cm.Reds)
    cmap4_2 = mpl.cm.ScalarMappable(norm=norm4, cmap=mpl.cm.Blues)
    
    sep1=lr1
    sep2=lr2
    sep3=lr3
    sep4=lr4
    sep5=lr5
    sep6=lr6
    sep7=lr7
    sep8=lr8

    ov1=overlap(sep1)
    ov2=overlap(sep2)
    ov3=overlap(sep3)
    ov4=overlap(sep4)
    ov5=overlap(sep5)
    ov6=overlap(sep6)
    ov7=overlap(sep7)
    ov8=overlap(sep8)
    
    plt.axes(xlabel='Zenith Distance (degrees)',ylabel='Dispersion (fraction of aperture)')
    plt.plot(zd,sep1,label='VIS LR B')
    plt.plot(zd,sep2,label='VIS LR V')
    plt.plot(zd,sep3,label='VIS LR R')
    plt.plot(zd,sep4,label='VIS LR All')
    plt.plot(zd,sep5,label='NR LR IY')
    plt.plot(zd,sep6,label='NR LR J')
    plt.plot(zd,sep7,label='NR LR H')
    plt.plot(zd,sep8,label='NR LR All')
    plt.ylim(0,2.5)
    plt.axhline(y=0.69,label='VIS fibre',linewidth=0.5,color='blue')
    plt.axhline(y=0.6,label='NIR fibre',linewidth=0.5,color='red')
    plt.legend(loc='upper left')

    #plt.savefig('ZD2.png')
    plt.show()

#     plt.axes(xlabel='Zenith Distance (degrees)',ylabel='Fractional Overlap')
# #    plt.plot(zd,ov1,label='Full Vis range')
#     plt.plot(zd,ov2,label='0.40-0.55')
#     plt.plot(zd,ov3,label='0.55-0.77')
#     plt.plot(zd,ov4,label='0.8-1.8')
#     plt.plot(zd,ov5,label='0.75-1.8')
#     plt.plot(zd,ov6,label='0.4-0.5')
#     plt.plot(zd,ov7,label='0.5-0.65')
#     plt.plot(zd,ov8,label='0.65-0.85')
#     plt.legend(loc='lower left')
#     plt.savefig('FO.png')
#     plt.show()
    return

