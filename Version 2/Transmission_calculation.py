
import math
import numpy as np
from astropy import units as u

from astropy.modeling.functional_models import Disk2D
from astropy.modeling.models import Gaussian2D
from astropy.modeling.models import Moffat2D

from astropy.io import fits


def analytical_gaussian(fibre_diameter,FWHMs,shifts,k_lim):
    """
    Calculates transmission of Gaussian PSF offset from a circular aperture
    See https://opg.optica.org/ao/fulltext.cfm?uri=ao-58-4-1048&id=404390
    Analytical solution featuring infinite sum

    INPUTS:
    fibre_diameter: float, astropy units
        diameter of the fibre/aperture
    FWHMs: array, astropy units
        array of FWHMs
    shifts: array, astropy units
        array of shifts
    k_lim: float
        term to carry out the infinite sum too; rapidly converges within typically 20 terms

    OUTPUTS:
    Returns:
    transmission: array
        array of transmissions
    """
    fibre_radius=fibre_diameter.value/2
    offset=shifts.value
    FWHMs=FWHMs.value

    prefactor=np.exp(-4*np.log(2)*offset**2/FWHMs**2)
    k_sum=0
    for k in range(0,k_lim):
        i_sum=0
        for i in range(0,k+1):
            i_sum=i_sum+(2**i*fibre_radius**(2*i)/math.factorial(i))*(2*np.log(2)/FWHMs**2)**i
        k_sum=k_sum+2**k*offset**(2*k)/math.factorial(k)*(2*np.log(2)/FWHMs**2)**k*(1-np.exp(-4*np.log(2)*fibre_radius**2/FWHMs**2)*i_sum)
    transmission=prefactor*k_sum
    return transmission

def calculate_FWHM(wavelength,airmass,median_FWHM,median_FWHM_lambda,kolb_factor):
    """
    Calculates FWHM of the monochromatic PSFs for different wavelengths and airmass

    INPUTS:
    wavelength: array, astropy units
        wavelength(s) to calculate the FWHM(s) at
    airmass: float
        airmass to calculate the FWHM at
    kolb_factor: boolean, True or False, default = False
        whether to use the kolb factor term of the varying PSF FWHM: see https://www.eso.org/observing/etc/doc/helpfors.html

    OUTPUTS:
    Returns:
    FWHM: array, astropy units
        FWHM of the monochromatic light depending on wavelength and airmass
    """
    if kolb_factor==True: #FWHM using kolb factor; smaller FWHM
        r0=0.1*median_FWHM.value**(-1)*(wavelength.value/500)**1.2*airmass**(-0.6)
        L0=46
        D=39 
        F_kolb=1/(1+300*(D/L0))-1
        FWHM=median_FWHM.value*airmass**(0.6)*(wavelength.value/500)**(-0.2)*np.sqrt(1+F_kolb*2.183*(r0/L0)**0.356)
    else: #FWHM without using kolb factor; larger FWHM
        FWHM=median_FWHM.value*(median_FWHM_lambda.value/wavelength.value)**(1/5)*(1/airmass)**(-3/5) #From FWHM proportional to x^3/5 * lamda^-1/5
    return FWHM * u.arcsec

def numerical_gaussian(fibre_diameter,FWHM,offset,scale):
    """
    Calculates transmission of Gaussian PSF offset from a circular aperture
    Numerical solution

    INPUTS:
    fibre_diameter: float, astropy units
        diameter of the fibre/aperture
    FWHM: float, astropy units
        FWHM of the PSF
    offset: float, astropy units
        offset of the PSF from the aperture
    scale: float
        scale of the numerical simulation, arcsec/pixel

    OUTPUTS:
    Returns:
    transmission: float
        transmission value
    """
    std = FWHM.value/(2*np.sqrt(2*np.log(2)))/scale
    x_pos=offset/scale

    gaussian_total= 2*np.pi*std**2 

    boundary=math.ceil(fibre_diameter.value/2/scale)

    x = np.arange(-boundary, boundary+1)
    y = np.arange(-boundary, boundary+1)
    x, y = np.meshgrid(x, y)

    Disk=Disk2D(1,0,0,fibre_diameter.value/2/scale)
    Disk_data=Disk(x,y)
    Gaussian=Gaussian2D(1,x_pos.value,0,std,std)
    Gaussian_data=Gaussian(x,y)

    convolved_data=Disk_data*Gaussian_data

    trans=sum(sum(convolved_data))/gaussian_total

    return trans

def numerical_moffat(fibre_diameter,FWHM,offset,scale,beta=2.5):
    """
    Calculates transmission of Moffat PSF offset from a circular aperture
    Numerical solution

    INPUTS:
    fibre_diameter: float, astropy units
        diameter of the fibre/aperture
    FWHM: float, astropy units
        FWHM of the PSF
    offset: float, astropy units
        offset of the PSF from the aperture
    scale: float
        scale of the numerical simulation, arcsec/pixel
    beta: float
        power index of the moffat equation, (also known as atmospheric scattering coefficient)

    OUTPUTS:
    Returns:
    transmission: float
        transmission value
    """
    alpha=FWHM.value/scale/(2*np.sqrt(2**(1/beta)-1))

    moffat_total=(np.pi*alpha**2)/(beta-1)

    x_pos=offset/scale  

    boundary=math.ceil(fibre_diameter.value/2/scale)

    x = np.arange(-boundary, boundary+1)
    y = np.arange(-boundary, boundary+1)
    x, y = np.meshgrid(x, y)

    Disk=Disk2D(1,0,0,fibre_diameter.value/2/scale)
    Disk_data=Disk(x,y)
    Moffat=Moffat2D(1,x_pos.value,0,alpha,beta)
    Moffat_data=Moffat(x,y)

    convolved_data=Disk_data*Moffat_data

    trans=sum(sum(convolved_data))/moffat_total
    return trans

def numerical_durham(diameter,wavelength,offset,axis_val):
    version=0 #1 IS FOR THE COMPRESSED PSFs; ~2x quicker with accuracy reduced to <1%
    file=fits.open("PSFs/GLAO_Median_{}nm_v2.fits".format(round(wavelength.value)))
    durham_data=file[version].data[axis_val]
    scale=file[version].header['scale']

    offset = abs(offset)
    
    fibre_boundary=math.ceil(diameter.value/2/scale)
    data_boundary=len(durham_data)

    x = np.arange(-fibre_boundary,fibre_boundary+1)
    y = np.arange(-fibre_boundary, fibre_boundary+1)
    x, y = np.meshgrid(x, y)

        
    disk=Disk2D(1,abs(int(offset.value/scale)-offset.value/scale),0,diameter.value/2/scale)
    disk_data=disk(x,y)

    resized_data=np.zeros([len(disk_data),len(disk_data)])

    durham_data=durham_data[int(data_boundary/2-fibre_boundary):int(data_boundary/2+fibre_boundary)+1,int(data_boundary/2-fibre_boundary+offset.value/scale):int(data_boundary/2+fibre_boundary+offset.value/scale)+1]
    resized_data[0:len(durham_data),0:len(durham_data[0])]=durham_data

    convolved=resized_data*disk_data
    trans=sum(sum(convolved))
    return trans

