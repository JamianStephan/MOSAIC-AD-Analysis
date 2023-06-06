import numpy as np
import matplotlib.pyplot as plt
import dispersion_functions as diff_func
import matplotlib as mpl

class AD_simulation:
    def __init__(self,**kwargs):
        self.input={}
        self.output={}
        
        self.config={
        'telescope_diameter':39, #m
        'wavefront_outer_scale':46, #m
        'median_seeing':.68, #arcsec
        'median_seeing_wl':.5, #um
        'simulation_scale':0.01, #arcsec/pixel, default is 0.01
        'relative_plate_PA_angle':0, #deg, relative angle of the plate/apertures and PA=0. For PA=0 along aperture semi major axis, set to 90 deg
        'centring':0.5, #either 0.5 for mid index, or HA index
        'latitude':-24.6272, #deg
        'temperature':10, #deg C
        'humidity':14.5, #%
        'pressure':750.0 #mBa
        }
        
        self.config.update(kwargs)
        
    def load_wavelengths(self,wavelengths):
        """

        """
        self.output['wavelengths']=np.array(wavelengths)

    def load_hour_angles(self,HA_range,declination):
        """
        
        """
        self.input['HA_range']=HA_range
        self.input['declination']=declination

        #latitude needs to be negative for now
        lat = float(self.config['latitude']) * np.pi/180
        dec = declination*np.pi/180
        
        #Need to check if the target is below the horizon for the given list of HA, and if so remove it.
        LHA_below_horizon=np.rad2deg(np.arccos(-np.tan(lat)*np.tan(dec)))/15 
        if str(LHA_below_horizon) != 'nan': 
            #print("Target goes below Horizon above/below HA of +/- %2.1fh" % (LHA_below_horizon))
            for val in HA_range.copy(): 
                if abs(val) > abs(LHA_below_horizon):
                    print("At HA %2.2fh, target goes below horizon - removing this from HA range" % (val))
                    HA_range.remove(val)        
        if dec > np.pi/2 + lat: #If the target has a too high declination, it will never be seen at Cerro Paranal
            print("Target always below Horizon")
            return

        airmasses=1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(np.array(HA_range)*2*np.pi/24))
        self.output['airmasses']=np.array(airmasses)

        self.input['ZA_range']=diff_func.airmass_to_ZA(airmasses)
        
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
        
    def load_PSFs(self,PSF_type="moffat",moffat_beta=2.5):
        """
        
        """
        self.input['PSF_type']=PSF_type
        all_PSFs=[]
        all_aligned_PSFs=[]
        
        zero_shifts=np.zeros(np.shape(self.output['shifts'][0]))

        for i in range(0,len(self.output['airmasses'])):
            PSFs=diff_func.make_PSFs(PSF_type,self.output['wavelengths'],self.output['shifts'][i],
                                                        self.output['airmasses'][i],self.input['major_axis'],self.config,beta=moffat_beta)
            aligned_PSFs=diff_func.make_PSFs(PSF_type,self.output['wavelengths'],zero_shifts,
                                                          self.output['airmasses'][i],self.input['major_axis'],self.config,beta=moffat_beta)
            all_PSFs.append(PSFs)
            all_aligned_PSFs.append(aligned_PSFs)

        self.output['PSFs']=all_PSFs
        self.output['aligned_PSFs']=all_aligned_PSFs
        
    def load_aperture(self,major_axis,aperture_type="hexagons",hexagon_radius=1):
        self.input['major_axis']=major_axis
        aperture=diff_func.make_aperture(aperture_type,major_axis,self.config,hexagon_radius=hexagon_radius)
        self.output['aperture']=aperture
        
    def calculate_integration_transmissions(self):
        convolved_arrays=self.output['PSFs']*self.output['aperture']
        convolved_aligned_arrays=self.output['aligned_PSFs']*self.output['aperture']
        self.output['PSFs_through_aperture']=convolved_arrays
        self.output['aligned_PSFs_through_aperture']=convolved_aligned_arrays

        raw_transmissions=np.sum(convolved_arrays,axis=(-1,-2))
        no_AD_transmissions=np.sum(convolved_aligned_arrays,axis=(-1,-2))
        relative_transmissions=raw_transmissions/no_AD_transmissions
        self.output['raw_transmissions']=raw_transmissions
        self.output['no_AD_transmissions']=no_AD_transmissions
        self.output['relative_transmissions']=relative_transmissions

        relative_integration_transmissions=np.mean(relative_transmissions,axis=0)
        no_AD_integration_transmissions=np.mean(no_AD_transmissions,axis=0)
        raw_integration_transmissions=np.mean(raw_transmissions,axis=0)
        self.output['raw_integration_transmissions']=raw_integration_transmissions
        self.output['no_AD_integration_transmissions']=no_AD_integration_transmissions
        self.output['relative_integration_transmissions']=relative_integration_transmissions

    def integration_plots(self,y_axis='relative',track_indexes=[0,-1]):
        """
        Function to illustrate simulation results
        1) Transmission vs wavelength curves for individual fibres and entire bundle
        2) Track plot of monochromatic spot PSFs on the aperture over an integration
        """
        plt.style.use('bmh')
        fig=plt.figure(figsize=[7,5])
        if y_axis=='relative':
            y_axis_data=self.output['relative_integration_transmissions']
            plt.axhline(y=1,label='No AD Transmission, {}'.format(self.input['PSF_type']),color='black',linestyle='--')
            plt.ylabel("Transmission Relative to No-AD")
        if y_axis=='raw':
            y_axis_data=self.output['raw_integration_transmissions']
            plt.ylabel("Raw Transmission")
            plt.plot(self.output['wavelengths'],self.output['no_AD_integration_transmissions'],color='black',linestyle='--',label='No AD Transmission, {}'.format(self.input['PSF_type']))
            
        plt.axvline(self.input['guide_waveref'],label="Guide = {}um".format(self.input['guide_waveref']),color='black',linestyle='--',linewidth=0.5)
        plt.axvline(self.input['aperture_waveref'],label="Aperture = {}um".format(self.input['aperture_waveref']),color='C0',linestyle='--',linewidth=0.5)
        plt.plot(self.output['wavelengths'],y_axis_data)
        plt.ylim(0,1.1)
        plt.xlabel("Wavelength (um)")
        plt.legend()
        
        fig, ax = plt.subplots(figsize=[5,5]) 
        weights = np.linspace(0, len(self.output['wavelengths'])-1,len(track_indexes))
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
        circle1 = plt.Circle((0, 0), self.input['major_axis']/2, color='black', fill=False, label='Major Axis')
        ax.add_patch(circle1)    
        plt.axvline(0,color='black',linestyle='--',linewidth=0.7,label="PA = {}".format(self.config['relative_plate_PA_angle']))
        plt.scatter(self.output['centre_shift'][0],self.output['centre_shift'][1],label='Guide = {}um'.format(self.input['guide_waveref']),color='black',marker='+')
        plt.xlim(-0.4,0.4)
        plt.ylim(-0.4,0.4)
        shifts=self.output['shifts']
        for count,i in enumerate(track_indexes):
            xs,ys=[],[]
            for o in range(0,len(shifts)):
                xs.append(shifts[o][i][0])
                ys.append(shifts[o][i][1]) 
            plt.plot(xs,ys,marker='x',color=cmap.to_rgba(weights[count]),label="{}um".format(round(self.output['wavelengths'][i],4)))
        plt.legend()
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")

    def load_ZA(self, ZA_range):
        """
        """
        airmasses=diff_func.ZA_to_airmass(ZA_range)
        self.output['airmasses']=airmasses
        self.input['ZA_range']=ZA_range
    
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
        convolved_arrays=self.output['PSFs']*self.output['aperture']
        convolved_aligned_arrays=self.output['aligned_PSFs']*self.output['aperture']
        self.output['PSFs_through_aperture']=convolved_arrays
        self.output['aligned_PSFs_through_aperture']=convolved_aligned_arrays

        raw_transmissions=np.sum(convolved_arrays,axis=(-1,-2))
        no_AD_transmissions=np.sum(convolved_aligned_arrays,axis=(-1,-2))
        relative_transmissions=raw_transmissions/no_AD_transmissions
        self.output['raw_transmissions']=raw_transmissions
        self.output['no_AD_transmissions']=no_AD_transmissions
        self.output['relative_transmissions']=relative_transmissions
        
    def snapshot_plots(self,y_axis='relative'):
        plt.style.use('bmh')
        fig=plt.figure(figsize=[7,5])
        if y_axis=='relative':
            y_data=self.output['relative_transmissions']
            plt.axhline(y=1,label='No AD Transmission, {}'.format(self.input['PSF_type']),color='black',linestyle='--')
            plt.ylabel("Transmission Relative to No-AD")
        if y_axis=='raw':
            y_data=self.output['raw_transmissions']
            plt.ylabel("Raw Transmission")
        
        plt.axvline(self.input['aperture_waveref'],label="Aperture = {}um".format(self.input['aperture_waveref']),color='C0',linestyle='--',linewidth=0.5)
        for count,trans in enumerate(y_data):
            plt.plot(self.output['wavelengths'],trans,label="ZA = {} deg".format(round(self.input['ZA_range'][count],1)))
        plt.ylabel("Transmission Relative to No-AD")
        plt.ylim(0,1.1)
        plt.xlabel("Wavelength (um)")
        plt.legend()   