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
#Parts adapted from Myriam Rodrigues code
class AD_analysis:
    def __init__(self):
        """Init values for the analysis"""
        #Loads config file for telescope parameters
        Config_tel = ConfigObj('./Architecture_parameters/Telescope_conf.ini')

        self.conditions = {} #Dictionary of conditions at Paranal and Instrument properties
        self.conditions['temperature']= float(Config_tel['EnvConditions']['AirTemperature']) * u.deg_C
        self.conditions['humidity']= float(Config_tel['EnvConditions']['AirHumidity']) * u.percent
        self.conditions['pressure']= float(Config_tel['EnvConditions']['AirPressure']) * u.mBa
        self.conditions['plate_scale'] = float(Config_tel['OpticalInterfaces']['Plate_Scale']) #* u.arcsec / u.mm, MOSAIC plate scale
        self.conditions['VIS_aperture_diameter']= 0.69 * u.arcsec #Diameter of VIS MOS aperture = Sampling * 3 = 0.234 * 3
        self.conditions['NIR_aperture_diameter']= 0.57 * u.arcsec #Diameter of IR MOS aperture = Sampling * 3 = 0.190 * 3
        self.conditions['median_FWHM']= 0.68 * u.arcsec #Median seeing at Paranal zenith, wavelength = 500nm, in arcsec
        self.conditions['median_FWHM_lambda']= 500 * u.nm #Wavelength of the median seeing at Paranal zenith, in nm       
        self.conditions['latitude']=-24.6272*u.deg #Paranal Latitude

        self.input = {} #Dictionary of all parameters used as inputs
        self.output = {} #Dictionary of all outputs used in the code

    def load_band(self,res,regime,min_band,max_band,sampling=1*u.nm):
        """
        Target PSF will be modelled as a series of monochromatic wavelengths
        This generates the monochromatic wavelengths to be used in the analysis based of a MOSAIC band

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
        regime, band, res

        Output dictionary:        
        aperture_diameter: float, in arcsec astropy units
            diameter of the aperture to use, gives _init_ values depending on inputs
        wavelengths: array, in nm astropy units
            array of the different monochromatic wavelengths to model
        """
        self.input['regime']=regime 
        self.input['res']=res
        self.input['band']=[min_band,max_band]

        Config_regime = ConfigObj('./Architecture_parameters/'+regime+'_channel_conf.ini') #Loads VIS or NIR parameters
        
        #Wave is sampled between min_band min wavelength and max_band max wavelength in intervals of "sampling" variable
        self.output['wavelengths'] = np.arange(int(Config_regime[res]['Bands'][min_band]['wave_min']),int(Config_regime[res]['Bands'][max_band]['wave_max'])+1,sampling.value) * u.nm

        #Store relevent aperture diameter depending on regime
        self.output['aperture_diameter']=self.conditions[regime+'_aperture_diameter']  

    def load_wave(self,aperture_diameter,start,end,sampling=1*u.nm):
        """
        Target PSF will be modelled as a series of monochromatic wavelengths
        This generates the monochromatic wavelengths to be used in the analysis

        INPUTS:
        aperture_diameter,
        start,
        end,
        sampling: float, in nm astropy units
                gap between each monochromatic wavelength
                
        OUTPUTS:
        Input dictionary:
        regime, band, res

        Output dictionary:        
        aperture_diameter: float, in arcsec astropy units
            diameter of the aperture to use, gives _init_ values depending on inputs
        wavelengths: array, in nm astropy units
            array of the different monochromatic wavelengths to model
        """
        self.input['regime']="N/A"
        self.input['res']="N/A"
        self.input['band']=["N/A","N/A"]
       
        #Wave is sampled between min_band min wavelength and max_band max wavelength in intervals of "sampling" variable
        self.output['wavelengths'] = np.arange(start,end,sampling.value) * u.nm

        #Store relevent aperture diameter depending on regime
        self.output['aperture_diameter']=aperture_diameter

    def load_airmasses(self,HA_range=[],ZA_range=[],airmasses=[],targ_dec=-25.3 * u.degree):
        """
        Need airmasses for analysis, 3 options:
        1) Calculate for a target declination at Cerro Paranal using a range of given hour angles
        2) Calculate using given angles from the zenith
        3) Give specific airmasses
        Chose by entering values into the list you want to use. 
        Don't use multiple, it's not designed for that.

        INPUTS:
        HA_range: list, integers, default = []
            range of hour angles to use
        ZA_range: list, in deg astropy units, default = []
            range of zenith angles to use
        airmasses: list, no units, default = []
            range of airmasses to use
        targ_dec: float, in astropy units, default = -25.3 degrees
           declination of target

        OUTPUTS:
        Input dictionary:
        HA_range, ZA_range, targ_dec

        Output dictionary:
        airmasses: numpy array of floats
            range of airmasses to use for anlaysis
        para_angles: numpy array of floats
            parallatic angle of the target at each HA
            
        """
        self.input['HA_range']=HA_range
        self.input['ZA_range']=ZA_range
        self.input['targ_dec']=targ_dec     
        
        if HA_range != []: #HA into airmasses
            lat = self.conditions['latitude'].to(u.rad).value 
            dec = targ_dec.to(u.rad).value 
            
            #Need to check if the target is below the horizon for the given list of HA, and if so remove it.
            LHA_below_horizon=np.rad2deg(np.arccos(-np.tan(lat)*np.tan(dec)))/15 
            if str(LHA_below_horizon) != 'nan': 
                print("Target goes below Horizon above/below HA of +/- %2.1fh" % (LHA_below_horizon))
                for val in HA_range.copy(): 
                    if abs(val) > abs(LHA_below_horizon):
                        print("At HA %2.2fh, target goes below horizon - removing this from HA range" % (val))
                        HA_range.remove(val) 
            if dec > np.pi/2 + lat: #If the target has a too high declination, it will never be seen at Cerro Paranal
                print("Target always below Horizon")
                return

            for HA in HA_range: #Calculates airmass for HAs (provided it is above Horizon)
                airmass=1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(HA*u.hour).rad))
                airmasses=np.append(airmasses,airmass)
            
            self.output['meridian_airmass'] = 1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(0*u.hour).rad)) #Lowest airmass the target will be at

            #Parallatic angles over the different HA
            para_angles=atm_diff.parallatic_angle(np.array(HA_range),targ_dec,self.conditions['latitude'])
            self.output['raw_para_angles']=np.array(para_angles) #actual PAs
    
        elif ZA_range != []: #ZA into airmasses
            for i in ZA_range:
                airmasses=np.append(airmasses,1/np.cos(np.deg2rad(i)))
            self.output['para_angles']=np.zeros(len(airmasses)) #PAs = 0, as these are snapshots
            
        elif airmasses != []:
            self.output['para_angles']=np.zeros(len(airmasses)) #PAs = 0, as these are snapshots

        self.output['airmasses']=np.array(airmasses)
        
    def calculate_shifts(self, guide_waveref, aperture_waveref, centring="mid HA",reposition = False, parallatic=True):
        """
        Calculates snapshots of the shifts of the monochromatic PSFs for given airmasses from load_airmasses
        Can either have the aperture at a fixed point, or at the centre of each snapshot

        INPUTS:
        guide_waveref: float, in astropy units, default = 0.537 microns
            wavelength the telescope is tracking on; this is the fixed point of the spectrum (doesn't matter if apertures are repositioned)
        aperture_waveref: float, in astropy units, default = 0.537 microns
            wavelength the apertures are centred on
        centring: string, "mid HA" or "mid airmass"  or an index
            FIX
        reposition: boolean, True or False, default = False
            whether to reposition the apertures each snapshot to the aperture_waveref wavelength, or keep them at the original position
            for snapshots, this = true. for integrations, this = false
        parallatic: boolean, True or False, default = True
            whether to include the parallatic angle effect in an integration

        OUTPUTS:
        Input dictionary:
        guide_waveref, aperture_waveref,reposition, centring index

        Output dictionary:
        shifts: numpy array, in astropy units
            shifts of the monochromatic PSFs for different airmasses. Form is [[airmass 1 shifts...][airmass 2 shifts..][...]...]
        """
        self.input['guide_waveref']=guide_waveref
        self.input['aperture_waveref']=aperture_waveref
        self.input['reposition']=reposition   
    
        airmasses=self.output['airmasses']
        wavelengths=self.output['wavelengths']

        shifts=[] 
        if reposition == True: #For every snapshot, aperture centre is repositioned to the current "aperture_waveref" wavelength
            for i in airmasses: #For each airmass, calculate AD shift
                #shift values out of the function are all relative to the guide wavelength
                centre_shift=atm_diff.diff_shift(aperture_waveref,i,guide_waveref,self.conditions) #shift of the aperture centre wavelength from guide wavelength
                shift=atm_diff.diff_shift(wavelengths,i,guide_waveref,self.conditions)-centre_shift #-centre shift, so shifts are relative to the current aperture centre wavelength
                shifts.append(shift)

        if reposition == False and parallatic == False: #For every snapshot, aperture centre is positioned to the first airmass' "aperture_waveref" wavelength
            centre_shift=atm_diff.diff_shift(aperture_waveref,airmasses[centring_index],guide_waveref,self.conditions) #shift of the original aperture centre wavelength from guide wavelength
            for i in airmasses: #for each airmass, calculate AD shift
                #shift values out of the function are all relative to the guide wavelength
                shift=atm_diff.diff_shift(wavelengths,i,guide_waveref,self.conditions)-centre_shift #-centre shift, so shifts are relative to the original aperture centre wavelength
                shifts.append(shift)

        if reposition == False and parallatic == True: #Same as above, but includes parallatic angles effect. Only works for HA 
                
            if centring == "mid HA":
                centring_index=int((len(airmasses)-1)/2)
            if type(centring) == int:
                centring_index=centring
            if type(centring) == int or centring == "mid HA":
                self.input['centred_on']=str(self.input['HA_range'][centring_index])+"h"
                centre_shift=atm_diff.diff_shift(aperture_waveref,airmasses[centring_index],guide_waveref,self.conditions) #shift of the original aperture centre wavelength from guide wavelength
                self.output['centre_shift']=centre_shift

                para_angles=self.output['raw_para_angles'].copy()
                for i in range(0,len(para_angles)): #change in PAs from centring index
                    para_angles[i]=para_angles[i]-self.output['raw_para_angles'][centring_index]
                
                self.input['centred_q']=self.output['raw_para_angles'][centring_index]

            if centring == "mid airmass":
                mid_airmass=(airmasses[0]+airmasses[-1])/2
                self.input['centred_on']="airmass "+str(round(mid_airmass,3))
                centre_shift=atm_diff.diff_shift(aperture_waveref,mid_airmass,guide_waveref,self.conditions) #shift of the original aperture centre wavelength from guide wavelength
                self.output['centre_shift']=centre_shift
                
                #need to turn airmass into parallatic angle
                mid_ZA=atm_diff.airmass_to_zenith_dist(mid_airmass)
                mid_HA=atm_diff.ZA_2_HA(mid_ZA,self.input['targ_dec'])
                mid_PA=atm_diff.parallatic_angle(mid_HA,self.input['targ_dec'],self.conditions['latitude'])

                para_angles=self.output['raw_para_angles'].copy()

                for i in range(0,len(para_angles)): #change in PAs from centring index
                    para_angles[i]=para_angles[i]-mid_PA.value
                
                self.input['centred_q']=mid_PA.value

            self.output['delta_para_angles']=para_angles
            shifts_no_para=[]            
            for count,i in enumerate(airmasses): #for each airmass, calculate AD shift
                #shift values out of the function are all relative to the guide wavelength
                shift_vals=atm_diff.diff_shift(wavelengths,i,guide_waveref,self.conditions)
                shifts_no_para.append(shift_vals.value-centre_shift.value)
                para_shift=np.sqrt(shift_vals**2+centre_shift**2-2*shift_vals*centre_shift*np.cos(para_angles[count])) #Uses cosine rule to evaluate the shift from the aperture centre
                #A^2=B^2+C^2-2BCcos(a), A = shift from aperture, B = aperture centre from guide, C = shift from guide, a = para angle change 
                
                #The para-shift sqrt only provides position values; define shifts from guide larger than the aperture centre from guide as positive, and smaller negative
                #This means there will be a jump from - to + values (e.g -0.3arcsec, to 0.3arcsec)
                for count,val in enumerate(shift_vals):
                    if val - centre_shift < 0:
                        para_shift[count]=-para_shift[count]
                shifts.append(para_shift)      
            self.output['shifts_no_para']=shifts_no_para*u.arcsec
            
        self.output['shifts']=np.array(shifts) * u.arcsec #Turn list into array with astropy units
             
    def make_aperture(self,aperture_type="circle",method="numerical moffat",scale=0.01,data_version=0):        
        """
        Generates the aperture for the evaluation of transmission (stores settings even if you dont need an aperture)

        INPUTS:
        aperture_type: string, one of "circle", "hexagons"
            wavelength the telescope is tracking on; this is the fixed point of the spectrum (doesn't matter if apertures are repositioned)
        method: string, one of "analytical gaussian", "numerical gaussian", "numerical moffat", "numerical durham"
            method to use in the transmission calculations
        scale: float in arcsec astropy units, default = 0.01 arcsec/pixel
            scale to carry out the transmission calculations - only matters for "numerican gaussian" and "numerical moffat"
        data_version: integer, one of 0 or 1, default = 0
            Uncompressed (=0) or compressed (=1) Durham PSFs, compressed is ~3x quicker but lower accuracy (~1% transmission)
        
        OUTPUTS:
        Input dictionary:
        aperture_type, method, scale, data_version
        
        Output dictionary:
        aperture_array: numpy array*
            generated aperture in array form
            
        *this will be a list of arrays if numerical durham method is used correponding to the different scale apertures
        """
        self.input['scale']=scale
        self.input['method']=method
        self.input['data_version']=data_version
        self.input['aperture_type']=aperture_type
        
        aperture_diameter=self.output['aperture_diameter']
        
        if method == "numerical durham":
            #Durham data has discrete PSFs with below wavelengths:
            PSF_wavelengths=[440,562,720,920,1202,1638]*u.nm
            aperture=[]
            band_centres=[]
            aperture=[[],[],[],[],[],[]] #apertures for each Durham PSF
            
            #To save time, we only need to make the apertures for the Durham PSF we're using, so find which ones we need
            for wavelength in self.output['wavelengths']:
                arg=abs(PSF_wavelengths.value-wavelength.value).argmin()
                band_centres.append(PSF_wavelengths[arg])
                
            #Each Durham PSF has a different scale, so an aperture needs to be generated for each
            for wavelength in set(band_centres):
                index=list(PSF_wavelengths.value).index(round(wavelength.value))
                file=fits.open("PSFs/GLAO_Median_{}nm_v2.fits".format(round(wavelength.value)))
                scale=file[data_version].header['scale']
                aperture[index]=trans_calc.make_aperture(aperture_type,scale,aperture_diameter)
                
            self.input['scale']="Durham"
            
        elif method == "analytical gaussian":
            #no aperture if we are using analytical gaussian method
            aperture=[]
        else:
            #make the aperture if numerical gaussian or numerical moffat
            aperture=trans_calc.make_aperture(aperture_type,scale,aperture_diameter)
            
        self.output['aperture_array']=aperture
        
    def calculate_transmissions(self,k_lim=50,FWHM_change=True,kolb_factor=True,beta=2.5,axis_val=24):     
        """
        Calculate the loaded waves' transmision using calculated shifts and aperture
        Can be done using an analytical gaussian method, a numerical gaussian/moffat method, or a numerical method with Durham PSFs

        INPUTS:
        k_lim: float, default=50
            number of terms to compute the sum to for the analytic transmission solution, 50 is a safe value
        FWHM_change: string, True or False
            whether to change the monochromatic FWHM with airmass and wavelength
        kolb_factor: boolean, True or False
            whether to use the kolb factor in FWHM change, as per https://www.eso.org/observing/etc/doc/helpfors.html
        beta: float, default=2.5
            moffat index to use
        axis_val: integer, 0-48, default = 24
            Durham PSF offset to use, centred = 24          

        OUTPUTS:74
        Input dictionary:
        FWHM_change, kolb_factor, k_lim, beta, axis_val
        
        Output dictionary:
        transmissions: numpy array of floats
            transmissions of the wave calculated through the chosen method. Form is [[airmass 1 transmissions...][airmass 2 transmissions...][...]...]
        """
        self.input['FWHM_change']=FWHM_change 
        self.input['kolb_factor']=kolb_factor
        self.input['k_lim']=k_lim
        self.input['beta']=beta
        self.input['axis_val']=axis_val
        
        scale=self.input['scale']
        method=self.input['method']
        data_version=self.input['data_version']
        aperture=self.output['aperture_array']

        wavelengths = self.output['wavelengths']
        airmasses=self.output['airmasses']
        shifts=self.output['shifts']
        aperture_diameter=self.output['aperture_diameter']
        median_FWHM=self.conditions['median_FWHM']
        median_FWHM_lambda=self.conditions['median_FWHM_lambda']

        if method=="numerical durham":
                PSF_wavelengths=[440,562,720,920,1202,1638]*u.nm #Discrete Durham PSFs available
                band_centres=[]
                transmissions=[]
                FWHMs=[]
                durham_PSFs=[[],[],[],[],[],[]]
                durham_scales=[[],[],[],[],[],[]]
                
                #Need to know which PSF to use for each generated monochromatic wavelength, aka the nearest
                for wavelength in wavelengths:
                    arg=abs(PSF_wavelengths.value-wavelength.value).argmin()
                    band_centres.append(PSF_wavelengths[arg])
               
                for wavelength in set(band_centres): #Only open needed Durham PSFs to save time
                    index=list(PSF_wavelengths.value).index(round(wavelength.value))
                    file=fits.open("PSFs/GLAO_Median_{}nm_v2.fits".format(round(wavelength.value)))
                    scale=file[data_version].header['scale']
                    durham_PSFs[index]=file[data_version].data[axis_val]
                    durham_scales[index]=scale

                for i in range(0,len(airmasses)): #transmission calculations
                    trans_list=[]
                    for o in range(0,len(wavelengths)):
                        index=list(PSF_wavelengths.value).index(round(band_centres[o].value))
                        trans=trans_calc.numerical_durham(aperture[index],durham_PSFs[index],shifts[i][o],durham_scales[index],data_version=data_version)
                        trans_list.append(trans)
                    transmissions.append(trans_list)   
                
        elif FWHM_change == True: 
            FWHMs = [] #List of the FWHM across the wave for each airmass, [[airmass 1 wave FWHM],[airmass 2 wave FWHM],....]
            for i in airmasses: #For each airmass, calculate the FWHM for each wavelength in the wave
                FWHMs.append(trans_calc.calculate_FWHM(wavelengths,i,median_FWHM,median_FWHM_lambda,kolb_factor))
            FWHMs=np.array(FWHMs) * u.arcsec
            
            if method=="analytical": #use analytical method
                transmissions=trans_calc.analytical_gaussian(aperture_diameter,FWHMs,shifts,k_lim) #Transmission of light into aperture using changing FWHM         
            elif method=="numerical gaussian": #Function does not take in arrays, so a loop is needed for each airmass and wavelength
                transmissions=[]
                for i in range(0,len(airmasses)): 
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wavelengths)): 
                        trans=trans_calc.numerical_gaussian(aperture,FWHMs[i][o],shifts[i][o],scale) 
                        trans_list.append(trans)
                    transmissions.append(trans_list) #Append this airmass transmissions' to the overall list     
            elif method=="numerical moffat": #Function does not take in arrays, so a loop is needed for each airmass and wavelength
                transmissions=[]
                for i in range(0,len(airmasses)): 
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wavelengths)): 
                        trans=trans_calc.numerical_moffat(aperture,FWHMs[i][o],shifts[i][o],scale,beta=beta) 
                        trans_list.append(trans) 
                    transmissions.append(trans_list) #Append this airmass transmissions' to the list
                    
        elif FWHM_change == False:
            FWHMs = [] #List of the FWHM across the wave for an airmass of one
            FWHM_oneairmass_val=trans_calc.calculate_FWHM(wavelengths,1,median_FWHM,median_FWHM_lambda,kolb_factor)
            for i in range(0,len(airmasses)):
                FWHMs.append(FWHM_oneairmass_val)

            if method=="analytical":
                transmissions=trans_calc.analytical_gaussian(aperture_diameter,median_FWHM,shifts,k_lim)  
            elif method=="numerical gaussian": #Loop needed
                transmissions=[]
                for i in range(0,len(airmasses)): 
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wavelengths)): 
                        trans=trans_calc.numerical_gaussian(aperture,FWHMs[i][o],shifts[i][o],scale)
                        trans_list.append(trans)
                    transmissions.append(trans_list) #Append this airmass transmissions' to the list
            elif method=="numerical moffat": #Loop needed
                transmissions=[]
                for i in range(0,len(airmasses)): 
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wavelengths)): 
                        trans=trans_calc.numerical_moffat(aperture,FWHMs[i][o],shifts[i][o],scale,beta=beta) 
                        trans_list.append(trans) 
                    transmissions.append(trans_list) 

        self.output['transmissions']=transmissions
        self.output['FWHMs']=FWHMs

