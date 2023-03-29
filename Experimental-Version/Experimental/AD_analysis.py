from configobj import ConfigObj
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from astropy.coordinates import Angle

from astroplan import Observer
import Atmospheric_diffraction as atm_diff
from astropy.io import fits

plt.style.use('bmh')
import Transmission_calculation as trans_calc

#Class for AD Analysis
#Adapted from Myriam Rodrigues code
class AD_analysis:
    def __init__(self):
        """Init values for the analysis"""
        #Loads config file for telescope parameters
        Config_tel = ConfigObj('./Architecture_parameters/Telescope_conf.ini')

        self.conditions = {} #Dictionary of environment conditions at Paranal
        self.conditions['temperature']= float(Config_tel['EnvConditions']['AirTemperature']) * u.deg_C
        self.conditions['humidity']= float(Config_tel['EnvConditions']['AirHumidity']) * u.percent
        self.conditions['pressure']= float(Config_tel['EnvConditions']['AirPressure']) * u.mBa
        self.plate_scale = float(Config_tel['OpticalInterfaces']['Plate_Scale']) #* u.arcsec / u.mm, MOSAIC plate scale
        self.latitude=-24.6272*u.deg #Paranal Latitude
        
        VIS_aperture_diameter=0.69 * u.arcsec #Diameter of VIS MOS aperture = Sampling * 3 = 234 * 3
        IR_aperture_diameter=0.57 * u.arcsec #Diameter of IR MOS aperture = Sampling * 3 = 190 * 3
        median_FWHM=0.68 * u.arcsec #median seeing at Paranal zenith, wavelength = 500nm, in arcsec
        median_FWHM_lambda = 500 * u.nm #wavelength of the median seeing at Paranal zenith, in nm

        self.input_parameters = {} #Dictionary of all parameters usd as inputs
        self.input_parameters['VIS_aperture_diameter']=VIS_aperture_diameter
        self.input_parameters['IR_aperture_diameter']=IR_aperture_diameter
        self.input_parameters['median_FWHM']=median_FWHM
        self.input_parameters['median_FWHM_lambda']=median_FWHM_lambda

        self.output_parameters = {} #Dictionary of all outputs used in the code

    def load_wave(self,res,regime,min_band,max_band,sampling=1*u.nm):
        """
        Target PSF will be modelled as a series of monochromatic wavelengths
        This generates the monochromatic wavelengths to be used in the analysis

        INPUTS:
        res: string, LR or HR
            which MOSAIC resolution the analysis will use
        regime: string, VIS or NIR
            which MOSAIC wavelength regime the analysis will use
        min_band: string, if VIS+HR: V or R. If VIS+LR: B,V, or R. If NIR+HR: IY or H. If NIR+LR: IY,J, or H
            band to use for the minimum wavelength
        max_band: string, same as above
            band to use for the maximum wavelength
        sampling: float, in nm astropy units
            gap between each monochromatic wavelength

        OUTPUTS:
        Input dictionary:
        regime, self.band, self.res: string
            used for labelling graphs during plot

        Output dictionary:        
        aperture_diameter: float, in astropy units
            diameter of the aperture to use, depends on _init_ values
        wave_wavelengths: array, in astropy units
            array of the different monochromatic wavelengths to model
        """
        self.input_parameters['regime']=regime #Store parameters in dictionary
        self.input_parameters['res']=res

        if min_band==max_band:
            self.input_parameters['band']=min_band
        else:
            self.input_parameters['band']="All"

        Config_regime = ConfigObj('./Architecture_parameters/'+regime+'_channel_conf.ini') #Loads VIS or NIR parameters
        #Wave is sampled between min_band min wavelength and max_band max wavelength in intervals of sampling variable
        self.output_parameters['wave_wavelengths'] = np.arange(int(Config_regime[res]['Bands'][min_band]['wave_min']),int(Config_regime[res]['Bands'][max_band]['wave_max'])+1,sampling.value) * u.nm

        if regime == 'VIS': #VIS and NIR apertures have different radii, stores appropriately
            self.output_parameters['aperture_diameter']=self.input_parameters['VIS_aperture_diameter']
        elif regime == 'NIR':
            self.output_parameters['aperture_diameter']=self.input_parameters['IR_aperture_diameter']
        return
    

    def load_airmasses(self,HA_range=[],ZA_range=[],airmasses=[],targ_dec=-25.3 * u.degree):
        """
        Need airmasses for analysis, 3 options:
        1) Calculated for a target declination at Cerro Paranal using a range of given hour angles
        2) Calculated using given angles from the zenith
        3) Calculated using given airmasses
        Chose by entering values into the list you want to use

        INPUTS:
        HA_range: list, in astropy units, default = []
            range of hour angles to use
        ZA_range: list, in astropy units, default = []
            range of zenith angles to use
        airmasses: list, no units, default = []
            range of airmasses to use
        targ_dec: float, in astropy units, default = -25.3 degrees
           declination of target

        OUTPUTS:
        Input dictionary:
        self.HA_range: array
            the hour angles used for the airmasses
        self.ZA_range: array
            the zenith angles used for the airmasses
        self.targ_dec: string
            declination of target, used for labelling plot

        Output dictionary:
        self.airmasses: array
            range of airmasses to use for anlaysis
        """
        if HA_range != [] and ZA_range != []: #Only does analysis for HA range or zenith angles. Why do you need both at once?
            print("Don't use both, use one or the other!")
            return

        if HA_range != []: #HA into airmasses
            lat = self.latitude.to(u.rad).value #Cerro Paranal Latitude
            dec = targ_dec.to(u.rad).value #Declination of target in radians
            
            #Need to check if the target is below the horizon for the given list of HA 
            LHA_below_horizon=np.rad2deg(np.arccos(-np.tan(lat)*np.tan(dec)))/15 #Local Hour Angle the target goes below the Horizon
            if str(LHA_below_horizon) != 'nan': #If there is an HA the target goes below Horizon, checks to see if any HA hours provided are for when the target is below horizon
                print("Target goes below Horizon above/below HA of +/- %2.1fh" % (LHA_below_horizon))
                for val in HA_range.copy(): #Check all HA angles given, and remove HA for which the target is below the Horizon
                    if abs(val) > abs(LHA_below_horizon):
                        print("At HA %2.2fh, target goes below horizon - removing this from HA range" % (val))
                        HA_range.remove(val) 
                        
            if dec > np.pi/2 + lat: #If the target has a too high declination, it will never be seen at Cerro Paranal
                print("Target always below Horizon")
                return

            for HA in HA_range: #Calculates airmass for HAs (provided it is above Horizon)
                airmass=1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(HA*u.hour).rad))
                airmasses=np.append(airmasses,airmass)
            
            self.output_parameters['meridian_airmass'] = 1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(0*u.hour).rad))

            para_angles=atm_diff.parallatic_angle(np.array(HA_range),targ_dec,self.latitude)
            para_angles_2=para_angles.copy()
            for i in range(0,len(para_angles)):
                para_angles_2[i]=para_angles[i]-para_angles[0]
            
            self.output_parameters['para_angles']=np.array(para_angles_2)

        elif ZA_range != []: #ZA into airmasses
            for i in ZA_range:
                airmasses=np.append(airmasses,1/np.cos(np.deg2rad(i)))
            self.output_parameters['para_angles']=np.zeros(len(airmasses))

        self.output_parameters['airmasses']=np.array(airmasses)
        self.input_parameters['HA_range']=HA_range
        self.input_parameters['ZA_range']=ZA_range
        self.input_parameters['targ_dec']=targ_dec     

    def calculate_shifts(self, guide_waveref=0.537 * u.micron, aperturecentre_waveref=0.537 * u.micron, reposition = False, parallatic=True):
        """
        Calculates snapshots of the shifts of the monochromatic PSFs for given airmasses from load_airmasses
        Can either have the aperture at a fixed point, or at the centre of each snapshot

        INPUTS:
        guide_waveref: float, in astropy units, default = 0.537 microns
            wavelength the telescope is tracking on; this is the fixed point of the spectrum (doesn't matter if apertures are repositioned)
        aperturecentre_waveref: float, in astropy units, default = 0.537 microns
            wavelength the apertures are centred on
        reposition: boolean, True or False, default = False
            whether to reposition the apertures each snapshot to the aperturecentre_waveref wavelength, or keep them at the original position

        OUTPUTS:
        Input dictionary:
        self.guide_waveref, self.aperturecentre_waveref: float, astropy units
            used for plotting
        self.reposition: boolean, True or False
            used for plotting

        Output dictionary:
        self.shifts: array, in astropy units
            shifts of the monochromatic PSFs for different airmasses. Form is [[airmass 1 shifts...][airmass 2 shifts..][...]...]
        """
        shifts=[] #AD Shifts
        self.input_parameters['guide_waveref']=guide_waveref
        self.input_parameters['aperturecentre_waveref']=aperturecentre_waveref
        self.input_parameters['reposition']=reposition      
    
        airmasses=self.output_parameters['airmasses']
        wave_wavelengths=self.output_parameters['wave_wavelengths']

        if reposition == True: #For every snapshot, aperture centre is repositioned to the current "aperturecentre_waveref" wavelength
            for i in airmasses: #for each airmass, calculate AD shift
                centre_shift=atm_diff.diff_shift(aperturecentre_waveref,i,guide_waveref,self.conditions) #shift of the aperture centre wavelength from guide wavelength
                shift=atm_diff.diff_shift(wave_wavelengths,i,guide_waveref,self.conditions)-centre_shift #shifts is relative to the current aperture centre wavelength
                shifts.append(shift)

        if reposition == False and parallatic == False: #For every snapshot, aperture centre is positioned to the first airmass' "aperturecentre_waveref" wavelength
            centre_shift=atm_diff.diff_shift(aperturecentre_waveref,airmasses[0],guide_waveref,self.conditions) #shift of the original aperture centre wavelength from guide wavelength
            for i in airmasses: #for each airmass, calculate AD shift
                shift=atm_diff.diff_shift(wave_wavelengths,i,guide_waveref,self.conditions)-centre_shift #shift is relative to original centre
                shifts.append(shift)

        #parallatic angles stuff:
        if reposition == False and parallatic == True: #For every snapshot, aperture centre is positioned to the first airmass' "aperturecentre_waveref" wavelength
            para_angles=self.output_parameters['para_angles']
            centre_shift=atm_diff.diff_shift(aperturecentre_waveref,airmasses[0],guide_waveref,self.conditions) #shift of the original aperture centre wavelength from guide wavelength
            for count,i in enumerate(airmasses): #for each airmass, calculate AD shift
                shift_vals=atm_diff.diff_shift(wave_wavelengths,i,guide_waveref,self.conditions)
                shift=np.sqrt(shift_vals**2+centre_shift**2-2*shift_vals*centre_shift*np.cos(para_angles[count])) #shift is relative to original centre
                shifts.append(shift)      
            
        self.output_parameters['shifts']=np.array(shifts) * u.arcsec #Turn list into array with astropy units
      
    def make_aperture(self,aperture_type="circle",method="numerical moffat",scale=0.01,data_version=0):        
        self.input_parameters['scale']=scale
        self.input_parameters['method']=method
        self.input_parameters['data_version']=data_version
        self.input_parameters['aperture_type']=aperture_type
        
        aperture_diameter=self.output_parameters['aperture_diameter']
        
        PSF_wavelengths=[440,562,720,920,1202,1638]*u.nm
        
        if method == "numerical durham":
            aperture=[]
            for wavelength in PSF_wavelengths:
                file=fits.open("PSFs/GLAO_Median_{}nm_v2.fits".format(round(wavelength.value)))
                scale=file[data_version].header['scale']
                aperture.append(trans_calc.make_aperture(aperture_type,scale,aperture_diameter))
                self.input_parameters['scale']="Durham"
        elif method == "analytical guassian":
            aperture=[]
        else:
            aperture=trans_calc.make_aperture(aperture_type,scale,aperture_diameter)
            
        self.output_parameters['aperture_array']=aperture
        
    def calculate_transmissions(self,k_lim=50,FWHM_change=True,kolb_factor=True,scale=0.01,beta=2.5,axis_val=24):     
        """
        Calculate the loaded waves' transmision using calculated shifts
        Can be done using an analytical gaussian method, or a numerical gaussian/moffat method
        aperture is currently modelled as a circular aperture

        INPUTS:
        k_lim: float, default=50
            number of terms to compute the sum to for the analytic transmission solution, 50 is a safe value
        FWHM_change: string, True or False
            whether to change the monochromatic FWHM with airmass and wavelength
        kolb_factor: boolean, True or False
            whether to use the kolb factor in FWHM change, as per https://www.eso.org/observing/etc/doc/helpfors.html
        method: string, "analytical", "numerical guassian", "numerical moffat", or "numerical durham"
            which method to use for calculating transmission
        scale: float, default=0.01
            scale to use in the numerical methods, arcsec/pixel
        beta: float, default=2.5
            moffat index to use
        axis_val: integer, 0-48, default = 24
            Durham PSF offset to use, centred = 24
        data_version: integer, 0 or 1, default = 0
            Uncompressed or compressed Durham PSFs, compressed is ~3x quicker but lower accuracy (~1% transmission)

        OUTPUTS:
        Input dictionary:
        self.FWHM_change, self.kolb_factor, self.k_lim, self.method: boolean, boolean, float, string
            used for plotting data later
        
        Output dictionary:
        self.wave_transmissions
            transmissions of the wave calculated through the chosen method. Form is [[airmass 1 transmissions...][airmass 2 transmissions...][...]...]
        """
        self.input_parameters['FWHM_change']=FWHM_change #Store all these parameters for later
        self.input_parameters['kolb_factor']=kolb_factor
        self.input_parameters['k_lim']=k_lim
        self.input_parameters['beta']=beta
        
        scale=self.input_parameters['scale']
        method=self.input_parameters['method']
        data_version=self.input_parameters['data_version']
        aperture=self.output_parameters['aperture_array']

        wave_wavelengths = self.output_parameters['wave_wavelengths']
        airmasses=self.output_parameters['airmasses']
        shifts=self.output_parameters['shifts']
        aperture_diameter=self.output_parameters['aperture_diameter']
        median_FWHM=self.input_parameters['median_FWHM']
        median_FWHM_lambda=self.input_parameters['median_FWHM_lambda']

        if method=="numerical durham":
                PSF_wavelengths=[440,562,720,920,1202,1638]*u.nm
                band_centres=[]
                wave_transmissions=[]
                FWHMs=[]
                
                durham_PSFs=[]
                durham_scales=[]
                
                for wavelength in PSF_wavelengths:
                    file=fits.open("PSFs/GLAO_Median_{}nm_v2.fits".format(round(wavelength.value)))
                    scale=file[data_version].header['scale']
                    durham_PSFs.append(file[data_version].data[axis_val])
                    durham_scales.append(scale)

                for wavelength in wave_wavelengths:
                    arg=abs(PSF_wavelengths.value-wavelength.value).argmin()
                    band_centres.append(PSF_wavelengths[arg])

                for i in range(0,len(airmasses)):
                    trans_list=[]
                    for o in range(0,len(wave_wavelengths)):
                        index=list(PSF_wavelengths.value).index(round(band_centres[o].value))
                        trans=trans_calc.numerical_durham(aperture[index],durham_PSFs[index],shifts[i][o],durham_scales[index],data_version=data_version)
                        trans_list.append(trans)
                    wave_transmissions.append(trans_list)   
                
        elif FWHM_change == True: #Is there a dependence of the FWHM on airmass and wavelength?
            FWHMs = [] #List of the FWHM across the wave for each airmass, [[airmass 1 wave FWHM],[airmass 2 wave FWHM],....]
            for i in airmasses: #For each airmass, calculate the FWHM for each wavelength in the wave
                FWHMs.append(trans_calc.calculate_FWHM(wave_wavelengths,i,median_FWHM,median_FWHM_lambda,kolb_factor))
            FWHMs=np.array(FWHMs) * u.arcsec
            
            if method=="analytical": #use analytical method
                wave_transmissions=trans_calc.analytical_gaussian(aperture_diameter,FWHMs,shifts,k_lim) #Transmission of light into aperture using changing FWHM       
                
            elif method=="numerical gaussian": #Use numerical gaussian method. Function does not take in arrays, so a loop is needed
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_gaussian(aperture,FWHMs[i][o],shifts[i][o],scale) #Transmission for given wavelength at given airmass
                        trans_list.append(trans)
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
                    
            elif method=="numerical moffat": #Use numerical moffat method. Function does not take in arrays, so a loop is needed
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_moffat(aperture,FWHMs[i][o],shifts[i][o],scale,beta=beta) #Transmission for given wavelength at given airmass
                        trans_list.append(trans) 
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
                    
        else:
            FWHMs=np.full((len(airmasses),len(wave_wavelengths)),median_FWHM) * u.arcsec
            
            if method=="analytical":
                wave_transmissions=trans_calc.analytical_gaussian(aperture_diameter,median_FWHM,shifts,k_lim) #Transmission of light into aperture using changing FWHM
                
            elif method=="numerical gaussian":
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_gaussian(aperture,FWHMs[i][o],shifts[i][o],scale) #Transmission for given wavelength at given airmass
                        trans_list.append(trans)
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
                    
            elif method=="numerical moffat":
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_moffat(aperture,FWHMs[i][o],shifts[i][o],scale,beta=beta) #Transmission for given wavelength at given airmass
                        trans_list.append(trans) 
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list

        self.output_parameters['wave_transmissions']=wave_transmissions
        self.output_parameters['FWHMs']=FWHMs

