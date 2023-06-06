from configobj import ConfigObj
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import dispersion_functions as diff_func
import matplotlib as mpl
from astropy.io import fits

class AD_simulation:
    def __init__(self,**kwargs):
        self.input={}
        self.output={}
        
        self.config={}
        imported_config=ConfigObj('conf.ini')
        for config_section in imported_config.values():
            for config_item in config_section.items():
                self.config[config_item[0]]=config_item[1]
        for kwarg in kwargs.items():
            self.config[kwarg[0]]=kwarg[1]
            
    def load_wavelengths(self,start,end,sampling,diameter):
        """

        """
        self.output['wavelengths']=np.arange(start,end,sampling)
        self.output['major_axis']=diameter
           
    def load_MOSAIC_band(self,band,sampling):
        """

        """
        self.input['band']=band

        wave_min,wave_max= float(self.config[band][0]),float(self.config[band][1])

        self.output['wavelengths'] = np.arange(wave_min,wave_max,sampling)   
        self.output['major_axis']=float(self.config[band[0:6]+"_major_axis"])
        
    def load_HA(self,HA_start,HA_end,declination):
        """
        
        """
        HA_range=np.linspace(HA_start,HA_end,int(self.config['HA_samples']))
        self.input['HA_range']=HA_range
        self.input['declination']=declination

        #latitude needs to be negative for now
        lat = float(self.config['latitude']) * np.pi/180
        dec = declination*np.pi/180
        
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

        airmasses=1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(HA_range*u.hour).rad))
        self.output['airmasses']=np.array(airmasses)
        
        self.input['ZAs']=diff_func.airmass_to_ZA(airmasses)

        para_angles=diff_func.parallatic_angle(np.array(HA_range),dec,lat)
        self.output['raw_para_angles']=np.array(para_angles) #actual PAs
    
    def calculate_integration_shifts(self, guide_waveref, aperture_waveref):
        self.input['guide_waveref']=guide_waveref
        self.input['aperture_waveref']=aperture_waveref

        airmasses=self.output['airmasses']
        wavelengths=self.output['wavelengths']

        #centring refers to the index of the hour angles at which we centre the aperture/guiding on a wavelength
        if float(self.config['centring']) == 0.5:
            centring_index=int((len(airmasses)-1)/2)
        else:
            centring_index=int(self.config['centring'])

        centre_shift=diff_func.diff_shift(aperture_waveref,airmasses[centring_index],guide_waveref,self.config) #shift of the original aperture centre wavelength from guide wavelength
        centring_q=self.output['raw_para_angles'][centring_index]

        raw_para_angles=self.output['raw_para_angles']
        para_angles=self.output['raw_para_angles'].copy()
        for i in range(0,len(para_angles)): #change in PAs from centring index
            para_angles[i]=para_angles[i]-self.output['raw_para_angles'][centring_index]

        shifts_para=[]
        phi=np.deg2rad(float(self.config['relative_plate_PA_angle']))
        for count,airmass in enumerate(airmasses): #for each airmass, calculate AD shift
            shift_vals=diff_func.diff_shift(wavelengths,airmass,guide_waveref,self.config)  
            airmass_shifts=[]

            for i in range(0,len(shift_vals)):
                x=(shift_vals[i])*np.sin(raw_para_angles[count])-centre_shift*np.sin(centring_q)
                y=(shift_vals[i])*np.cos(raw_para_angles[count])-centre_shift*np.cos(centring_q)
                airmass_shifts.append([x*np.cos(phi)-y*np.sin(phi),y*np.cos(phi)+x*np.sin(phi)])
                
            shifts_para.append(airmass_shifts)

        self.output['shifts']=np.array(shifts_para)
        centre_shift_para=[-centre_shift*np.sin(centring_q),-centre_shift*np.cos(centring_q)]
        centre_shift_para=[centre_shift_para[0]*np.cos(phi)-centre_shift_para[1]*np.sin(phi),
                           centre_shift_para[1]*np.cos(phi)+centre_shift_para[0]*np.sin(phi)]
        self.output['centre_shift']=centre_shift_para
    
    def load_PSFs(self):
        """
        
        """
        PSF_wavelength_options=[.440,.562,.720,.920,1.202,1.638] #Discrete Durham PSFs available
        PSF_ZA_options=[0,15,30,45,60]
        wavelengths=self.output['wavelengths']
        ZAs=self.input['ZAs']
        
        PSF_wavelengths=[]
        PSF_ZAs=[]

        #Need to know which PSF to use for each generated monochromatic wavelength, aka the nearest
        for wavelength in wavelengths:
            arg=abs(PSF_wavelength_options-wavelength).argmin()
            PSF_wavelengths.append(PSF_wavelength_options[arg])
        
        for ZA in ZAs:
            arg=abs(np.array(PSF_ZA_options)-ZA).argmin()
            PSF_ZAs.append(PSF_ZA_options[arg])

        needed_PSFs={}
        needed_scales={}
        
        for wavelength in set(PSF_wavelengths): #Only open needed Durham PSFs to save time
            for ZA in set(PSF_ZAs):
                file=fits.open("PSFs/GLAO_Median_{}nm_zen{}deg.fits".format(round(wavelength*1000),round(ZA)))
                PSF_scale=file[0].header['scale']
                PSF_data=file[0].data[24]
                needed_PSFs[str(wavelength)+"nm_"+str(ZA)+"deg_PSF"]=PSF_data
                needed_scales[str(wavelength)+"nm_"+str(ZA)+"deg_scale"]=PSF_scale
                
        scales=[]
        PSFs=[]
        aligned_PSFs=[]
        for airmass_count,airmass_shifts in enumerate(self.output['shifts']):
            wavelength_PSFs=[]
            wavelength_scales=[]
            wavelength_aligned_PSFs=[]
            for wavelength_count,wavelength_shift in enumerate(airmass_shifts):
                ZA=PSF_ZAs[airmass_count]
                wavelength=PSF_wavelengths[wavelength_count]
                PSF_data=needed_PSFs[str(wavelength)+"nm_"+str(ZA)+"deg_PSF"]
                PSF_scale=needed_scales[str(wavelength)+"nm_"+str(ZA)+"deg_scale"]
                self.bug=[PSF_data,wavelength_shift,self.output['major_axis'],PSF_scale]
                cropped_PSF_data=diff_func.crop_durham_PSF(PSF_data,wavelength_shift,self.output['major_axis'],PSF_scale)
                aligned_PSF_data=diff_func.crop_durham_PSF(PSF_data,[0,0],self.output['major_axis'],PSF_scale)
                
                wavelength_scales.append(PSF_scale)
                wavelength_PSFs.append(cropped_PSF_data)
                wavelength_aligned_PSFs.append(aligned_PSF_data)
            scales.append(wavelength_scales)
            PSFs.append(wavelength_PSFs)
            aligned_PSFs.append(wavelength_aligned_PSFs)
                
        self.output['PSFs']=PSFs
        self.output['aligned_PSFs']=aligned_PSFs
        self.output['scales']=scales

    def load_aperture(self,aperture_type="hexagons"):
        apertures=[]
        for scale in set(self.output['scales'][0]):
            self.config['scale']=scale
            aperture=diff_func.make_aperture(aperture_type,self.input['band'],self.output['major_axis'],self.config)
            apertures.append(aperture)
        self.output['apertures']=apertures
           
    def calculate_integration_transmissions(self):
        #need to do the process for each PSF in the datacube with the correct aperture
        #either loop or do a data cube with respective aperture for each PSF
        convolved_PSFs=[]
        convolved_aligned_PSFs=[]
        for o in range(0,len(self.output['PSFs'])):
            airmass_convolved_PSFs=[]
            airmass_convolved_aligned_PSFs=[]
            for count,PSF in enumerate(self.output['PSFs'][o]):
                for i in range(0,len(self.output['apertures'])):
                    if self.output['scales'][o][count]==list(set(self.output['scales'][o]))[i]:
                        aperture=self.output['apertures'][i]    
                convolved_PSF=PSF*aperture
                convolved_aligned_PSF=self.output['aligned_PSFs'][o][count]*aperture
                airmass_convolved_PSFs.append(sum(sum(convolved_PSF)))
                airmass_convolved_aligned_PSFs.append(sum(sum(convolved_aligned_PSF)))
            convolved_PSFs.append(airmass_convolved_PSFs)
            convolved_aligned_PSFs.append(airmass_convolved_aligned_PSFs)
            
            raw_transmissions=np.array(convolved_PSFs)
            no_AD_transmissions=np.array(convolved_aligned_PSFs)
            relative_transmissions=raw_transmissions/no_AD_transmissions
            self.output['raw_transmissions']=raw_transmissions
            self.output['no_AD_transmissions']=no_AD_transmissions
            self.output['relative_transmissions']=relative_transmissions

            relative_integration_transmissions=np.mean(relative_transmissions,axis=0)
            raw_integration_transmissions=np.mean(raw_transmissions,axis=0)
            self.output['raw_integration_transmissions']=raw_integration_transmissions
            self.output['relative_integration_transmissions']=relative_integration_transmissions
            
    def integration_plots(self):
            """
            Function to illustrate simulation results
            1) Transmission vs wavelength curves for individual fibres and entire bundle
            2) Track plot of monochromatic spot PSFs on the aperture over an integration
            """
            if self.input['band'][3:6]=="NIR":
                colour='red'
            else:
                colour='blue'
            plt.style.use('bmh')
            fig=plt.figure(figsize=[7,5])
            plt.axhline(y=1,label='No AD Transmission, {}'.format("numerical moffat"),color='black',linestyle='--')
            plt.axvline(self.input['guide_waveref'],label="Guide = {}um".format(self.input['guide_waveref']),color='black',linestyle='--',linewidth=0.5)
            plt.plot(self.output['wavelengths'],self.output['relative_integration_transmissions'],label="Aperture = {}um".format(self.input['aperture_waveref']),color=colour)
            plt.ylabel("Transmission Relative to No-AD")
            plt.ylim(0,1.1)
            plt.xlabel("Wavelength (um)")
            plt.legend()
            
            fig, ax = plt.subplots(figsize=[5,5]) 
            weights = np.linspace(0, len(self.output['wavelengths'])-1,4)
            norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
            circle1 = plt.Circle((0, 0), self.output['major_axis']/2, color='black', fill=False, label='~Aperture')
            ax.add_patch(circle1)    
            plt.axvline(0,color='black',linestyle='--',linewidth=0.7,label="PA = {}".format(self.config['relative_plate_PA_angle']))
            plt.scatter(self.output['centre_shift'][0],self.output['centre_shift'][1],label='Guide = {}um'.format(self.input['guide_waveref']),color='black',marker='+')
            plt.xlim(-0.4,0.4)
            plt.ylim(-0.4,0.4)
            shifts=self.output['shifts']
            for i in weights:
                xs,ys=[],[]
                for o in range(0,len(shifts)):
                    xs.append(shifts[o][int(i)][0])
                    ys.append(shifts[o][int(i)][1]) 
                plt.plot(xs,ys,marker='x',color=cmap.to_rgba(int(i)),label="{}um".format(round(self.output['wavelengths'][int(i)],4)))
            plt.legend()
            plt.xlabel("x (arcsec)")
            plt.ylabel("y (arcsec)")

    def load_ZA(self, ZA_vals):
        """
        """
        airmasses=diff_func.ZA_to_airmass(ZA_vals)
        self.output['airmasses']=airmasses
        self.input['ZAs']=ZA_vals
    
    def calculate_snapshot_shifts(self,aperture_waveref):
        self.input['aperture_waveref']=aperture_waveref
    
        airmasses=self.output['airmasses']
        wavelengths=self.output['wavelengths']
        
        shifts=[]
        for count,airmass in enumerate(airmasses):
            centre_shift=diff_func.diff_shift(aperture_waveref,airmass,1,self.config)
            airmass_shifts=diff_func.diff_shift(wavelengths,airmass,1,self.config)-centre_shift
            airmass_shifts=np.append(np.resize(airmass_shifts,(len(airmass_shifts),1)),np.zeros((len(airmass_shifts),1)),axis=-1)
            shifts.append(airmass_shifts)
        self.output['shifts']=shifts

    def calculate_snapshot_transmissions(self):
        #need to do the process for each PSF in the datacube with the correct aperture
        #either loop or do a data cube with respective aperture for each PSF
        convolved_PSFs=[]
        convolved_aligned_PSFs=[]
        for o in range(0,len(self.output['PSFs'])):
            airmass_convolved_PSFs=[]
            airmass_convolved_aligned_PSFs=[]
            for count,PSF in enumerate(self.output['PSFs'][o]):
                for i in range(0,len(self.output['apertures'])):
                    if self.output['scales'][o][count]==list(set(self.output['scales'][o]))[i]:
                        aperture=self.output['apertures'][i]    
                convolved_PSF=PSF*aperture
                convolved_aligned_PSF=self.output['aligned_PSFs'][o][count]*aperture
                airmass_convolved_PSFs.append(sum(sum(convolved_PSF)))
                airmass_convolved_aligned_PSFs.append(sum(sum(convolved_aligned_PSF)))
            convolved_PSFs.append(airmass_convolved_PSFs)
            convolved_aligned_PSFs.append(airmass_convolved_aligned_PSFs)
            
            raw_transmissions=np.array(convolved_PSFs)
            no_AD_transmissions=np.array(convolved_aligned_PSFs)
            relative_transmissions=raw_transmissions/no_AD_transmissions
            self.output['raw_transmissions']=raw_transmissions
            self.output['no_AD_transmissions']=no_AD_transmissions
            self.output['relative_transmissions']=relative_transmissions

    def snapshot_plots(self):
        plt.style.use('bmh')
        fig=plt.figure(figsize=[7,5])
        plt.axhline(y=1,label='No AD Transmission, {}'.format("numerical moffat"),color='black',linestyle='--')
        plt.axvline(self.input['aperture_waveref'],label="Aperture = {}um".format(self.input['aperture_waveref']),color='black',linestyle='--',linewidth=0.5)
        for count,trans in enumerate(self.output['relative_transmissions']):
            plt.plot(self.output['wavelengths'],trans,label="ZA = {} deg".format(self.input['ZAs'][count]))
        plt.ylabel("Transmission Relative to No-AD")
        plt.ylim(0,1.1)
        plt.xlabel("Wavelength (um)")
        plt.legend()   
