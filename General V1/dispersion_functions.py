import math
import numpy as np
from astropy.modeling.models import Moffat2D
from astropy.modeling.functional_models import Disk2D
from matplotlib.path import Path

def calculate_FWHM(wavelength,airmass,config):
    D = float(config['telescope_diameter'])
    L0 = float(config['wavefront_outer_scale'])
    median_seeing = float(config['median_seeing'])
    median_seeing_wl = float(config['median_seeing_wl'])
       
    r0=0.1*median_seeing**(-1)*(wavelength/median_seeing_wl)**1.2*airmass**(-0.6)  
    F_kolb=1/(1+300*(D/L0))-1
    
    FWHM_atm=median_seeing*airmass**(0.6)*(wavelength/median_seeing_wl)**(-0.2)*np.sqrt(1+F_kolb*2.183*(r0/L0)**0.356)
    FWHM_dl=0.212*wavelength/D
    
    FWHM_total=np.sqrt(FWHM_atm**2+FWHM_dl**2)

    return FWHM_total

def make_PSFs(PSF_type,wavelengths,offsets,airmass,major_axis,config,beta=2.5):
    wavelengths,offsets = np.array(wavelengths),np.array(offsets)
    scale=float(config['simulation_scale'])
    pixel_radius=math.ceil(major_axis/2/scale) #radius of aperture in pixels

    FWHMs = calculate_FWHM(wavelengths,airmass,config)

    x = np.arange(0, pixel_radius*2)
    y = np.arange(0, pixel_radius*2)
    x, y = np.meshgrid(x, y)
          
    PSFs=np.zeros((len(wavelengths),pixel_radius*2,pixel_radius*2))
    if PSF_type=="moffat":
        for count in range(0,len(wavelengths)):
            alpha=FWHMs[count]/scale/(2*np.sqrt(2**(1/beta)-1))
            moffat_total=(np.pi*alpha**2)/(beta-1)
            x_pos=offsets[count][0]/scale  
            y_pos=offsets[count][1]/scale
            
            Moffat=Moffat2D(1,x_pos+pixel_radius-0.5,y_pos+pixel_radius-0.5,alpha,beta)
            Moffat_data=Moffat(x,y)
            PSFs[count]=Moffat_data/moffat_total
            
    if PSF_type=="gaussian":
        print("not done yet")
        
    return PSFs

def parallatic_angle(HA,dec,lat):
    HA=np.array(HA)*2*np.pi/24
    q = np.arctan2(np.sin(HA),(np.cos(dec)*np.tan(lat)-np.sin(dec)*np.cos(HA)))
    return q

def diff_shift(wave, airmass, atm_ref_wav,config):
    Lambda0 = atm_ref_wav
    wave = wave

    T = float(config["temperature"])+273.15
    HR = float(config["humidity"])/100
    P = float(config["pressure"])
    
    ZD_deg = airmass_to_ZA(airmass)
    ZD = np.deg2rad(ZD_deg)

    # saturation pressure Ps (millibars)
    PS = -10474.0 + 116.43*T - 0.43284*T**2 + 0.00053840*T**3

    # water vapour pressure
    Pw = HR * PS
    # dry air pressure
    Pa = P - Pw

    #dry air density
    Da = (1 + Pa * (57.90*1.0e-8 - 0.0009325/T + 0.25844/T**2)) * Pa/T
    #water vapour density
    Dw = (1 + Pw * (1 + 3.7 * 1E-4 * Pw) * (- 2.37321 * 1E-3 + 2.23366/T
                                            - 710.792/T**2
                                            + 77514.1/T**3)) * Pw/T

    S0 = 1.0/Lambda0
    S = 1.0/wave

    N0_1 = (1.0E-8*((2371.34+683939.7/(130.0-S0**2)+4547.3/(38.9-S0**2))*Da
            + (6487.31+58.058*S0**2-0.71150*S0**4+0.08851*S0**6)*Dw))
    N_1 = 1.0E-8*((2371.34+683939.7/(130.0-S**2)+4547.3/(38.9-S**2))*Da
                  + (6487.31+58.058*S**2-0.71150*S**4+0.08851*S**6)*Dw)

    DR = np.tan(ZD)*(N0_1-N_1)

    return DR*3600*180/np.pi

def airmass_to_ZA(airmass):
    return np.rad2deg(np.arccos(1/airmass))

def ZA_to_airmass(ZA):
    return 1/np.cos(np.deg2rad(ZA))

def cube_direction(direction):
    direction_vectors=[[+1,0],[+1,-1],[0,-1],[-1,0],[-1,+1],[0,+1]]
    return direction_vectors[direction]
def cube_add(hex,vec):
    return [hex[0]+vec[0],hex[1]+vec[1]]
def cube_neighbor(cube,direction):
    return cube_add(cube,cube_direction(direction))
def cube_scale(hex, factor):
    return [hex[0]*factor,hex[1]*factor]
def cube_ring(center, radius):
    results = []
    hex = cube_add(center,
                        cube_scale(cube_direction(4), radius))
    for i in range(0,6):
        for j in range(0,radius):
            results.append(hex)
            hex = cube_neighbor(hex, i)
    return results
def cube_spiral(center, radius):
    results = [center]
    for k in range(1,radius+1):
        ring_results=cube_ring(center, k)
        for i in ring_results:
            results.append(i)
    return results

def make_aperture(type,major_axis,config,hexagon_radius=1):
    scale=float(config['simulation_scale'])
    
    pixel_radius=math.ceil(major_axis/2/scale) #radius of aperture in pixels
    if type == "circle":
    
        x = np.arange(0, pixel_radius*2)
        y = np.arange(0, pixel_radius*2)
        x, y = np.meshgrid(x, y)
 
        Disk=Disk2D(1,pixel_radius-0.5,pixel_radius-0.5,major_axis/2/scale)
        aperture=Disk(x,y)    

        return aperture
    
    if type == "hexagons":
        sampling = major_axis/(2*hexagon_radius-1)/scale
        triangle_side=sampling*np.sqrt(3)/3
        aperture_centre=[pixel_radius-0.5,pixel_radius-0.5]
        
        centres_array=(cube_spiral([0,0],hexagon_radius-1))
        q_vector=np.array([sampling/2,sampling/np.sqrt(3)*3/2])
        r_vector=np.array([sampling,0])
        
        centres_xy_coords=[]
        for array_val in centres_array:
            xy_vals=np.array(array_val[0]*q_vector+array_val[1]*r_vector)
            centres_xy_coords.append(xy_vals)
  
        apertures=np.zeros((len(centres_array),pixel_radius*2,pixel_radius*2))
        for count,vals in enumerate(centres_xy_coords):
            centre=[vals[1]+aperture_centre[0],vals[0]+aperture_centre[1]]
            P1=[centre[0]+triangle_side*np.cos(np.pi*1/3),centre[1]+triangle_side*np.sin(np.pi/3)]
            P2=[centre[0]+triangle_side,centre[1]]
            P3=[centre[0]+triangle_side*np.cos(np.pi/3),centre[1]-triangle_side*np.sin(np.pi/3)]
            P4=[centre[0]-triangle_side*np.cos(np.pi/3),centre[1]-triangle_side*np.sin(np.pi/3)]
            P5=[centre[0]-triangle_side,centre[1]]
            P6=[centre[0]-triangle_side*np.cos(np.pi/3),centre[1]+triangle_side*np.sin(np.pi/3)]
            polygon=[P1,P2,P3,P4,P5,P6]
            height=pixel_radius*2
            width=pixel_radius*2
            poly_path=Path(polygon)

            x, y = np.mgrid[:height, :width]
            coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))

            mask = poly_path.contains_points(coors)
            mask=mask.reshape(height, width)
            
            apertures[count]=mask
            
        aperture=np.sum(apertures,axis=0)
        return aperture
        
def line(A,B):
    m=(A[1]-B[1])/(A[0]-B[0])
    c=A[1]-m*A[0] 
    return m,c

def crop_durham_PSF(PSF,offset,major_axis,scale):
    boundary=math.ceil(major_axis/2/scale)

    data_boundary=len(PSF)
    resized_data=np.zeros([boundary*2,boundary*2])

    ymin=math.ceil(data_boundary/2-boundary-offset[1]/scale)
    ymax=math.ceil(data_boundary/2+boundary-offset[1]/scale)
    xmin=math.ceil(data_boundary/2-boundary-offset[0]/scale)
    xmax=math.ceil(data_boundary/2+boundary-offset[0]/scale)
        
    ymin_old=ymin
    xmin_old=xmin

    if ymin < 0:
        ymin=1
    if xmin < 0:
        xmin=1
    if xmax > len(PSF):
        xmax=len(PSF)
    if ymax > len(PSF):
        ymax=len(PSF)    
    
    if ymax > 0 and xmax > 0 and xmax - xmin_old > 0 and ymax - ymin_old > 0: 
        resized_data[ymin-ymin_old:ymax-ymin_old,xmin-xmin_old:xmax-xmin_old]=PSF[ymin:ymax,xmin:xmax]
    
    return resized_data