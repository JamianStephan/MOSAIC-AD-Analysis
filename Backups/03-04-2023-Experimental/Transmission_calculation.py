import math
import numpy as np
from astropy import units as u

from astropy.modeling.models import Gaussian2D
from astropy.modeling.models import Moffat2D
from astropy.modeling.functional_models import Disk2D

# This module contains functions needed to calculate transmissions from PSF shifts
def analytical_gaussian(aperture_diameter,FWHMs,shifts,k_lim):
    """
    Calculates transmission of Gaussian PSF offset from a circular aperture
    See https://opg.optica.org/ao/fulltext.cfm?uri=ao-58-4-1048&id=404390
    Analytical solution featuring infinite sum

    INPUTS:
    aperture_diameter: float, arcsec astropy units
        diameter of the aperture/aperture
    FWHMs: array, astropy units
        array of FWHMs
    shifts: array, astropy units
        array of shifts
    k_lim: float
        term to carry out the infinite sum too; rapidly converges within typically ~20 terms

    OUTPUTS:
    Returns:
    transmission: array
        array of transmissions
    """
    aperture_radius=aperture_diameter.to(u.arcsec).value/2
    offset=shifts.to(u.arcsec).value
    FWHMs=FWHMs.to(u.arcsec).value

    prefactor=np.exp(-4*np.log(2)*offset**2/FWHMs**2)
    k_sum=0
    for k in range(0,k_lim):
        i_sum=0
        for i in range(0,k+1):
            i_sum=i_sum+(2**i*aperture_radius**(2*i)/math.factorial(i))*(2*np.log(2)/FWHMs**2)**i
        k_sum=k_sum+2**k*offset**(2*k)/math.factorial(k)*(2*np.log(2)/FWHMs**2)**k*(1-np.exp(-4*np.log(2)*aperture_radius**2/FWHMs**2)*i_sum)
    transmission=prefactor*k_sum
    return transmission

def calculate_FWHM(wavelength,airmass,median_FWHM,median_FWHM_lambda,kolb_factor=True):
    """
    Calculates FWHM of the monochromatic PSFs for different wavelengths and airmass

    INPUTS:
    wavelength: array, nm astropy units
        wavelength(s) to calculate the FWHM(s) at
    airmass: float
        airmass to calculate the FWHM at
    kolb_factor: boolean, True or False, default = False
        whether to use the kolb factor/outer scale term of the varying PSF FWHM: see https://www.eso.org/observing/etc/doc/helpfors.html

    OUTPUTS:
    Returns:
    FWHM: array, arcsec astropy units
        FWHM of the monochromatic light depending on wavelength and airmass
    """
    if kolb_factor==True: #FWHM using kolb factor; smaller FWHM
        r0=0.1*median_FWHM.to(u.arcsec).value**(-1)*(wavelength.to(u.nm).value/median_FWHM_lambda.to(u.nm).value)**1.2*airmass**(-0.6)
        L0=46
        D=39 
        F_kolb=1/(1+300*(D/L0))-1
        FWHM=median_FWHM.to(u.arcsec).value*airmass**(0.6)*(wavelength.to(u.nm).value/median_FWHM_lambda.to(u.nm).value)**(-0.2)*np.sqrt(1+F_kolb*2.183*(r0/L0)**0.356)
    else: #FWHM without using kolb factor; larger FWHM
        FWHM=median_FWHM.to(u.arcsec).value*(median_FWHM_lambda.to(u.nm).value/wavelength.to(u.nm).value)**(1/5)*(1/airmass)**(-3/5) #From FWHM proportional to x^3/5 * lamda^-1/5
    return FWHM * u.arcsec

def numerical_durham(aperture,PSF,offset,scale,axis_val=24,data_version=0):
    """
    Calculates transmission of Durham PSF offset from a circular aperture
    Numerical solution

    INPUTS:
    aperture_diameter: float, arcsec astropy units
        diameter of the aperture/aperture
    wavelength: array, nm astropy units, from [440,562,720,920,1202,1638]nm
        wavelength of the Durham PSF to use, corresponding to old band centres
    offset: float, arcsec astropy units
        offset of the PSF from the aperture
    axis_val: float, from [0-48]
        GLAO axis offset for the PSF. 25 is perfectly centred
    data_version: float, from [0,1]
        which Durham PSF to use; 0 is the original, 1 is the compressed data to ~0.01 arcsec

    OUTPUTS:
    Returns:
    transmission: float
        transmission value
    """
    durham_data=PSF

    offset = abs(offset.to(u.arcsec).value)
    aperture_boundary=(len(aperture)-1)/2 #radius of aperture in pixels
    data_boundary=len(durham_data)

    resized_data=np.zeros([len(aperture),len(aperture)])
    durham_data=durham_data[int(data_boundary/2-aperture_boundary):int(data_boundary/2+aperture_boundary)+1,int(data_boundary/2-aperture_boundary+offset/scale):int(data_boundary/2+aperture_boundary+offset/scale)+1]
    resized_data[0:len(durham_data),0:len(durham_data[0])]=durham_data
    convolved=resized_data*aperture
    
    trans=sum(sum(convolved))

    return trans

def numerical_moffat(aperture,FWHM,offset,scale,beta=2.5):
    """
    Calculates transmission of Moffat PSF offset from a circular aperture
    Numerical solution

    INPUTS:
    aperture_diameter: float, arcsec astropy units
        diameter of the aperture/aperture
    FWHM: float, arcsec astropy units
        FWHM of the PSF
    offset: float, arcsec astropy units
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
    alpha=FWHM.to(u.arcsec).value/scale/(2*np.sqrt(2**(1/beta)-1))

    moffat_total=(np.pi*alpha**2)/(beta-1)

    x_pos=offset/scale  

    boundary=(len(aperture)-1)/2

    x = np.arange(-boundary, boundary+1)
    y = np.arange(-boundary, boundary+1)
    x, y = np.meshgrid(x, y)

    Moffat=Moffat2D(1,x_pos.to(u.arcsec).value,0,alpha,beta)
    Moffat_data=Moffat(x,y)

    convolved_data=aperture*Moffat_data

    trans=sum(sum(convolved_data))/moffat_total
    return trans

def numerical_gaussian(aperture,FWHM,offset,scale):
    """
    Calculates transmission of Gaussian PSF offset from a circular aperture
    Numerical solution

    INPUTS:
    aperture_diameter: float, arcsec astropy units
        diameter of the aperture/aperture
    FWHM: float, arcsec astropy units
        FWHM of the PSF
    offset: float, arcsec astropy units
        offset of the PSF from the aperture
    scale: float
        scale of the numerical simulation, arcsec/pixel

    OUTPUTS:
    Returns:
    transmission: float
        transmission value
    """
    std = FWHM.to(u.arcsec).value/(2*np.sqrt(2*np.log(2)))/scale
    x_pos=offset/scale

    gaussian_total= 2*np.pi*std**2 

    boundary=(len(aperture)-1)/2

    x = np.arange(-boundary, boundary+1)
    y = np.arange(-boundary, boundary+1)
    x, y = np.meshgrid(x, y)

    Gaussian=Gaussian2D(1,x_pos.value,0,std,std)
    Gaussian_data=Gaussian(x,y)

    convolved_data=aperture*Gaussian_data

    trans=sum(sum(convolved_data))/gaussian_total

    return trans

def line(A,B):
    m=(A[1]-B[1])/(A[0]-B[0])
    c=A[1]-m*A[0] 
    return m,c

def make_aperture(type,scale,major_axis,hex_rotation=0):
    boundary=math.ceil(major_axis.to(u.arcsec).value/2/scale) #radius of aperture in pixels
    if type == "circle":
    
        x = np.arange(-boundary, boundary+1)
        y = np.arange(-boundary, boundary+1)
        x, y = np.meshgrid(x, y)
 
        Disk=Disk2D(1,0,0,major_axis.value/2/scale)
        aperture=Disk(x,y)    

        return aperture
    
    if type == "hexagons":
        sampling = major_axis.value/3/scale
        aperture_array=np.zeros([boundary*2+1,boundary*2+1])

        triangle_side=sampling*np.sqrt(3)/3
        core = 2 * triangle_side * np.cos(np.pi/4)
        aperture_centre=[boundary,boundary]
        alpha = hex_rotation
        
        centre_0=aperture_centre
        centre_1=[centre_0[0]+sampling*np.cos(np.pi/2-alpha),centre_0[1]+sampling-sampling*(1-np.sin(np.pi/2-alpha))]
        centre_2=[centre_0[0]+np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.cos(np.pi/6-alpha),centre_0[1]+np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.sin(np.pi/6-alpha)]
        centre_3=[centre_0[0]+np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.cos(np.pi/6+alpha),centre_0[1]-np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.sin(np.pi/6+alpha)]
        centre_4=[centre_0[0]-sampling*np.cos(np.pi/2-alpha),centre_0[1]-sampling+sampling*(1-np.sin(np.pi/2-alpha))]
        centre_5=[centre_0[0]-np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.cos(np.pi/6-alpha),centre_0[1]-np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.sin(np.pi/6-alpha)]
        centre_6=[centre_0[0]-np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.cos(np.pi/6+alpha),centre_0[1]+np.sqrt((triangle_side*3/2)**2+(sampling/2)**2)*np.sin(np.pi/6+alpha)]

        centres=[centre_0,centre_1,centre_2,centre_3,centre_4,centre_5,centre_6]

        for centre in centres:
            if alpha == 0:
                P1=[centre[0]+triangle_side*np.cos(np.pi*1/3-alpha),centre[1]+triangle_side*np.sin(np.pi/3-alpha)]
                P2=[centre[0]+triangle_side*np.cos(alpha),centre[1]-triangle_side*np.sin(alpha)]
                P3=[centre[0]+triangle_side*np.cos(np.pi/3+alpha),centre[1]-triangle_side*np.sin(np.pi/3+alpha)]
                P4=[centre[0]-triangle_side*np.cos(np.pi/3-alpha),centre[1]-triangle_side*np.sin(np.pi/3-alpha)]
                P5=[centre[0]-triangle_side*np.cos(-alpha),centre[1]-triangle_side*np.sin(-alpha)]
                P6=[centre[0]-triangle_side*np.cos(np.pi/3+alpha),centre[1]+triangle_side*np.sin(np.pi/3+alpha)]

                L12_m,L12_c=line(P1,P2)
                L23_m,L23_c=line(P2,P3)
                L34_m,L34_c=line(P3,P4)
                L45_m,L45_c=line(P4,P5)
                L56_m,L56_c=line(P5,P6)
                L61_m,L61_c=line(P6,P1)
                           
                for y in range(0,len(aperture_array)):
                    for x in range(0,len(aperture_array)):           
                        if x < centre_0[0] + triangle_side * 2 and x > centre_0[0] - triangle_side * 2 and y < centre_0[1] + sampling and y > centre_0[1] - sampling: 
                            aperture_array[y][x]=1         
                        elif y < L61_m*x + L61_c and y > L34_m*x + L34_c and y < L12_m*x + L12_c and y > L23_m*x + L23_c and y > L45_m*x + L45_c and y < L56_m*x + L56_c:
                            aperture_array[y][x]=1  
                            
            elif alpha != np.pi/6 and alpha != -np.pi/6:
                P1=[centre[0]+triangle_side*np.cos(np.pi*1/3-alpha),centre[1]+triangle_side*np.sin(np.pi/3-alpha)]
                P2=[centre[0]+triangle_side*np.cos(alpha),centre[1]-triangle_side*np.sin(alpha)]
                P3=[centre[0]+triangle_side*np.cos(np.pi/3+alpha),centre[1]-triangle_side*np.sin(np.pi/3+alpha)]
                P4=[centre[0]-triangle_side*np.cos(np.pi/3-alpha),centre[1]-triangle_side*np.sin(np.pi/3-alpha)]
                P5=[centre[0]-triangle_side*np.cos(-alpha),centre[1]-triangle_side*np.sin(-alpha)]
                P6=[centre[0]-triangle_side*np.cos(np.pi/3+alpha),centre[1]+triangle_side*np.sin(np.pi/3+alpha)]

                L12_m,L12_c=line(P1,P2)
                L23_m,L23_c=line(P2,P3)
                L34_m,L34_c=line(P3,P4)
                L45_m,L45_c=line(P4,P5)
                L56_m,L56_c=line(P5,P6)
                L61_m,L61_c=line(P6,P1)
                
                for y in range(0,len(aperture_array)):
                    for x in range(0,len(aperture_array)):
                        if y > centre_0[1] - core and y < centre_0[1] + core and x > centre_0[1] - core and x < centre_0[1] + core:
                            aperture_array[y][x]=1
                        elif y < L61_m*x + L61_c and y > L34_m*x + L34_c and y < L12_m*x + L12_c and y > L23_m*x + L23_c and y > L45_m*x + L45_c and y < L56_m*x + L56_c:
                            aperture_array[y][x]=1
                                
            elif alpha == np.pi/6 or alpha == - np.pi/6:
                P1=[centre[0]+triangle_side*np.cos(np.pi/3-alpha),centre[1]+triangle_side*np.sin(np.pi/3-alpha)]
                P2=[centre[0]+triangle_side*np.cos(alpha),centre[1]-triangle_side*np.sin(alpha)]
                P3=[centre[0]+triangle_side*np.cos(np.pi/3+alpha),centre[1]-triangle_side*np.sin(np.pi/3+alpha)]
                P4=[centre[0]-triangle_side*np.cos(np.pi/3-alpha),centre[1]-triangle_side*np.sin(np.pi/3-alpha)]
                P5=[centre[0]-triangle_side*np.cos(-alpha),centre[1]-triangle_side*np.sin(-alpha)]
                P6=[centre[0]-triangle_side*np.cos(np.pi/3+alpha),centre[1]+triangle_side*np.sin(np.pi/3+alpha)]
                
                L23_m,L23_c=line(P2,P3)
                L34_m,L34_c=line(P3,P4)
                L56_m,L56_c=line(P5,P6)
                L61_m,L61_c=line(P6,P1)
                
                for y in range(0,len(aperture_array)):
                    for x in range(0,len(aperture_array)):
                        if y < centre_0[1] + triangle_side * 2 and y > centre_0[1] - triangle_side * 2 and x < centre_0[0] + sampling and x > centre_0[0] - sampling: 
                            aperture_array[y][x]=1    
                        elif y < L61_m*x + L61_c and y > L34_m*x + L34_c and y > L23_m*x + L23_c and  y < L56_m*x + L56_c and x > centre[0] - sampling/2 and x < centre[0] + sampling/2:
                            aperture_array[y][x]=1   
                             
        return aperture_array