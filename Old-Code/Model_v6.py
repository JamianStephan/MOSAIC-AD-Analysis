
from configobj import ConfigObj
import subprocess
import os, sys
import math
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Markdown
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.coordinates import SkyCoord,Angle
from astropy.time import Time
from astroplan import Observer
from Atmospheric_diffraction import *

import math
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
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
        
        VIS_fibre_diameter=0.69 * u.arcsec #Diameter of VIS MOS fibre
        IR_fibre_diameter=0.6 * u.arcsec #Diameter of IR MOS fibre
        median_FWHM=0.68 * u.arcsec #median seeing at Paranal zenith, wavelength = 500nm, in arcsec!
        median_FWHM_lambda = 500 * u.nm #wavelength of the median seeing at Paranal zenith, in nm!

        self.input_parameters = {} #Dictionary of all parameters usd as inputs
        self.input_parameters['VIS_fibre_diameter']=VIS_fibre_diameter
        self.input_parameters['IR_fibre_diameter']=IR_fibre_diameter
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
        sampling: float, in nm
            gap between each monochromatic wavelength

        OUTPUTS:
        Input dictionary:
        regime, self.band, self.res: string
            used for labelling graphs during plot

        Output dictionary:        
        fibre_diameter: float, in astropy units
            diameter of the fibre to use, depends on _init_ values
        wave_wavelengths: array, in astropy units
            array of the different monochromatic wavelengths to model
        """
        self.input_parameters['regime']=regime #Store parameters in dictionary
        self.input_parameters['res']=res

        if min_band==max_band:
            self.input_parameters['band']=min_band
        else:
            self.input_parameters['band']="All"
            self.input_parameters['band']="All"

        Config_regime = ConfigObj('./Architecture_parameters/'+regime+'_channel_conf.ini') #Loads VIS or NIR parameters
        #Wave is sampled between min_band min wavelength and max_band max wavelength in intervals of sampling variable
        self.output_parameters['wave_wavelengths'] = np.arange(int(Config_regime[res]['Bands'][min_band]['wave_min']),int(Config_regime[res]['Bands'][max_band]['wave_max'])+1,sampling.value) * u.nm

        if regime == 'VIS': #VIS and NIR fibres have different radii, stores appropriately
            self.output_parameters['fibre_diameter']=self.input_parameters['VIS_fibre_diameter']
        elif regime == 'NIR':
            self.output_parameters['fibre_diameter']=self.input_parameters['IR_fibre_diameter']
        return
    
    def HA_2_ZA(self,HA,dec):
        lat = np.deg2rad(-24.6272) #Cerro Paranal Latitude
        dec = dec.to(u.rad).value #Declination of target in radians 
        alt=np.arcsin(np.cos(np.deg2rad(np.array(HA)*15))*np.cos(lat)*np.cos(dec)+np.sin(lat)*np.sin(dec)) #altitude of target at HA
        ZA=90-np.rad2deg(alt) #ZA of target at HA
        return ZA


    def ZA_2_HA(self,ZA,dec):
        lat = np.deg2rad(-24.6272) #Cerro Paranal Latitude
        dec = dec.to(u.rad).value #Declination of target in radians   
        alt = np.deg2rad(90 - ZA)
        HA = np.arccos((np.sin(alt)-np.sin(lat)*np.sin(dec))/(np.cos(lat)*np.cos(dec)))
        return np.rad2deg(HA)/15

    def load_airmasses(self,HA_range=[],ZA_range=[],targ_dec=-25.3 * u.degree):
        """
        Need airmasses for analysis, 2 options:
        1) Calculated for a target declination at Cerro Paranal using a range of given hour angles
        2) Calculated using given angles from the zenith
        Chose by entering values into the list you want to use

        INPUTS:
        HA_range: list, in astropy units, default = []
            range of hour angles to use
        ZA_range: list, in astropy units, default = []
            range of zenith angles to use
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
        airmasses = np.array([]) #airmasses stored to array

        self.input_parameters['ZA_range']=ZA_range
        self.input_parameters['targ_dec']=targ_dec     

        if HA_range != [] and ZA_range != []: #Only does analysis for HA range or zenith angles. Why do you need both at once?
            print("Don't use both, use one or the other!")
            return

        if HA_range != []: #If HA values have been entered, use them
            print("HA used")
            lat = np.deg2rad(-24.6272) #Cerro Paranal Latitude
            dec = targ_dec.to(u.rad).value #Declination of target in radians
            
            #Need to check if the target is below the horizon for the given list of HA 
            LHA_below_horizon=np.rad2deg(np.arccos(-np.tan(lat)*np.tan(dec)))/15 #Local Hour Angle the target goes below the Horizon
            if str(LHA_below_horizon) != 'nan': #If there is an HA the target goes below Horizon, checks to see if any HA hours provided are for when the target is below horizon
                print("Target goes below Horizon above/below HA of +/- %2.1fh" % (LHA_below_horizon))
                for val in HA_range.copy(): #Check all HA angles given
                    if abs(val) > abs(LHA_below_horizon):
                        print("At HA %2.2fh, target goes below horizon - removing this from HA range" % (val))
                        HA_range.remove(val) #Removes HAs for which the target is below the Horizon

            if dec > np.pi/2 + lat: #If the target has a too high declination, it will never be seen at Cerro Paranal
                print("Target always below Horizon")
                return

            for HA in HA_range: #Calculates airmass for given HAs (provided it is above Horizon)
                airmass=1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(HA*u.hour).rad))
                airmasses=np.append(airmasses,airmass)
            
            self.output_parameters['meridian_airmass'] = 1/(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(Angle(0*u.hour).rad))

        elif ZA_range != []: #If zenith angles have been entered, use them
            print("Zenith Angles Used")
            for i in ZA_range:
                airmasses=np.append(airmasses,1/np.cos(np.deg2rad(i)))

        self.output_parameters['airmasses']=airmasses
        self.input_parameters['HA_range']=HA_range

    def calculate_snapshifts(self, guide_waveref=0.537 * u.micron,plot=False, fibrecentre_waveref=0.537 * u.micron, reposition = False):
        """
        Calculates snapshots of the shifts of the monochromatic PSFs for given airmasses from load_airmasses
        Can either have the fibre at a fixed point, or at the centre of each snapshot

        INPUTS:
        guide_waveref: float, in astropy units, default = 0.537 microns
            wavelength the telescope is tracking on; this is the fixed point of the spectrum (doesn't matter if fibres are repositioned)
        fibrecentre_waveref: float, in astropy units, default = 0.537 microns
            wavelength the fibres are centred on
        reposition: boolean, True or False, default = False
            whether to reposition the fibres each snapshot to the fibrecentre_waveref wavelength, or keep them at the original position
        plot: boolean, True of False, default = False
            plot shift graphs, 1) wavelength vs displacement for different HA/ZA, 2) displacement vs HA/ZA for different wavelengths

        OUTPUTS:
        Input dictionary:
        self.guide_waveref, self.fibrecentre_waveref: float, astropy units
            used for plotting
        self.reposition: boolean, True or False
            used for plotting

        Output dictionary:
        self.shifts: array, in astropy units
            shifts of the monochromatic PSFs for different airmasses. Form is [[airmass 1 shifts...][airmass 2 shifts..][...]...]
        """
        shifts=[] #AD Shifts

        self.input_parameters['guide_waveref']=guide_waveref
        self.input_parameters['fibrecentre_waveref']=fibrecentre_waveref
        self.input_parameters['reposition']=reposition

    
        airmasses=self.output_parameters['airmasses']
        wave_wavelengths=self.output_parameters['wave_wavelengths']

        if reposition == True: #For every snapshot, fibre centre is repositioned to the current "fibrecentre_waveref" wavelength
            #print("Reposition = True")
            for i in airmasses: #for each airmass, calculate AD shift
                centre_shift=Atmospheric_diffraction(fibrecentre_waveref,i,guide_waveref,self.conditions) #shift of the fibre centre wavelength from guide wavelength
                shift=Atmospheric_diffraction(wave_wavelengths,i,guide_waveref,self.conditions)-centre_shift #shifts is relative to the current fibre centre wavelength
                shifts.append(shift)

        elif reposition == False: #For every snapshot, fibre centre is positioned to the first airmass' "fibrecentre_waveref" wavelength
            #print("Reposition = False")
            centre_shift=Atmospheric_diffraction(fibrecentre_waveref,airmasses[0],guide_waveref,self.conditions) #shift of the original fibre centre wavelength from guide wavelength
            for i in airmasses: #for each airmass, calculate AD shift
                shift=Atmospheric_diffraction(wave_wavelengths,i,guide_waveref,self.conditions)-centre_shift #shift is relative to original centre
                shifts.append(shift)

        self.output_parameters['shifts']=np.array(shifts) * u.arcsec #Turn list into array with astropy units

        if plot==True and self.input_parameters['HA_range'] != []: #If plot is True and HA values are used, plot the 2 graphs
            HA_range=self.input_parameters['HA_range']
            HA_to_ZA_range=self.HA_2_ZA(HA_range,self.input_parameters['targ_dec'])
            print(HA_to_ZA_range)
            #1) wavelength vs displacement for different HA 
            centre_shift=Atmospheric_diffraction(fibrecentre_waveref,airmasses[0],guide_waveref,self.conditions)
            T_arc = lambda T_mm: T_mm * self.plate_scale
            T_mm = lambda T_arc: T_arc / self.plate_scale
            fig, ax = plt.subplots(figsize=(10,7))
            ax2 = ax.secondary_yaxis("right", functions=(T_arc, T_mm))
            if reposition == False: #If repositioned, wavelength reference centre is meaningless as relative position changes
                plt.axhline(-centre_shift.value,linewidth=0.5,color='red', label='Wavelength Reference Centre')
            for i in range(0,len(airmasses)):
                plt.plot(wave_wavelengths,shifts[i],label="HA = %2.2fh" %(HA_range[i]))
            plt.axhline(0,linewidth=0.8,color='black',label='Fibre Centre')
            plt.axhline(self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',label='Fibre Boundary',linestyle='--')
            plt.axhline(-self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',linestyle='--')
            plt.title('Wavelength Reference %s, Fibre Centre on %s, Dec = %2.2f deg' %(guide_waveref,fibrecentre_waveref,self.input_parameters['targ_dec'].value))
            plt.legend(loc='best')

            ax.set_ylabel('Displacement from Fibre Centre (arcsec)')
            ax2.set_ylabel('Displacement from Fibre Centre (mm)')
            ax.set_xlabel('Wavelength (nm)')

            #2) Displacement vs HA for different wavelengths
            to_HA = lambda ZA_vals: self.ZA_2_HA(ZA_vals,self.input_parameters['targ_dec'])
            to_ZA = lambda HA_vals: self.HA_2_ZA(HA_vals,self.input_parameters['targ_dec'])
            fig, ax = plt.subplots(figsize=(10,7))
            ax2 = ax.secondary_yaxis("right", functions=(to_ZA, to_HA))
            if reposition == False: #If repositioned, wavelength reference centre is meaningless as relative position changes
                plt.axvline(-centre_shift.value,linewidth=0.5,color='red',label='Wavelength Reference Centre')
            xvals=np.linspace(-1,1,len(wave_wavelengths))
            c=np.tan(xvals)
            for i in range(0,len(shifts)):
                yvals=np.full(len(shifts[i]),HA_range[i])
                plt.scatter(shifts[i],yvals,c=c)
                norm = plt.Normalize(wave_wavelengths.value.min(), wave_wavelengths.value.max())
                points = np.array([shifts[i], yvals]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(wave_wavelengths.value)
                lc.set_linewidth(2)
                line=ax.add_collection(lc)
            fig.colorbar(line)
            plt.axvline(0,linewidth=0.7,color='black',label='Fibre Centre')
            plt.axvline(self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',label='Fibre Boundary',linestyle='--')
            plt.axvline(-self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',linestyle='--')
            plt.xlabel('Displacement from Fibre Centre (arcsec)')
            plt.ylabel('Hour Angle (h)')
            ax2.set_ylabel("Zenith Angle (deg)")
            plt.yticks(np.linspace(min(HA_range),max(HA_range),len(HA_range)))
            plt.legend()
            plt.title('Wavelength Reference %s, Fibre Centre on %s, Dec = %2.2f deg' %(guide_waveref,fibrecentre_waveref,self.input_parameters['targ_dec'].value))    

        if plot==True and self.input_parameters['ZA_range'] != []: #If plot is True and Zenith Angle values are used, plot the 2 graphs
            ZA_range=self.input_parameters['ZA_range']


            #1) wavelength vs displacement for different HA 
            centre_shift=Atmospheric_diffraction(fibrecentre_waveref,airmasses[0],guide_waveref,self.conditions)
            T_arc = lambda T_mm: T_mm * self.plate_scale
            T_mm = lambda T_arc: T_arc / self.plate_scale
            fig, ax = plt.subplots(figsize=(10,7))
            ax2 = ax.secondary_yaxis("right", functions=(T_arc, T_mm))
            if reposition == False: #If repositioned, wavelength reference centre is meaningless as relative position changes
                plt.axhline(-centre_shift.value,linewidth=0.5,color='red', label='Wavelength Reference Centre')
            for i in range(0,len(airmasses)):
                plt.plot(wave_wavelengths,shifts[i],label="ZA = %2.2f deg" %(ZA_range[i]))
            plt.axhline(0,linewidth=0.8,color='black',label='Fibre Centre')
            plt.axhline(self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',label='Fibre Boundary',linestyle='--')
            plt.axhline(-self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',linestyle='--')
            plt.title('Wavelength Reference %s, Fibre Centre on %s' %(guide_waveref,fibrecentre_waveref))
            plt.legend(loc='best')
            ax.set_ylabel('Displacement from Fibre Centre (arcsec)')
            ax2.set_ylabel('Displacement from Fibre Centre (mm)')
            ax.set_xlabel('Wavelength (nm)')

            #2) Displacement vs ZA for different wavelengths
            fig, ax = plt.subplots(figsize=(10,7))

            if reposition == False: #If repositioned, wavelength reference centre is meaningless as relative position changes
                plt.axvline(-centre_shift.value,linewidth=0.5,color='red',label='Wavelength Reference Centre')
            xvals=np.linspace(-1,1,len(wave_wavelengths))
            c=np.tan(xvals)
            for i in range(0,len(shifts)):
                yvals=np.full(len(shifts[i]),ZA_range[i])
                plt.scatter(shifts[i],yvals,c=c)
                norm = plt.Normalize(wave_wavelengths.value.min(), wave_wavelengths.value.max())
                points = np.array([shifts[i], yvals]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(wave_wavelengths.value)
                lc.set_linewidth(2)
                line=ax.add_collection(lc)
            fig.colorbar(line)
            plt.axvline(0,linewidth=0.8,color='black',label='Fibre Centre')
            plt.axvline(self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',label='Fibre Boundary',linestyle='--')
            plt.axvline(-self.output_parameters['fibre_diameter'].value/2,linewidth=0.8,color='black',linestyle='--')
            plt.xlabel('Displacement from Fibre Centre (arcsec)')
            plt.ylabel('Zenith Angle (deg)')
            plt.yticks(np.linspace(min(ZA_range),max(ZA_range),len(ZA_range)))
            plt.legend()
            plt.title('Wavelength Reference %s, Fibre Centre on %s' %(guide_waveref,fibrecentre_waveref))      
        
    #Below is not documented                
    def calculate_snaptransmissions(self,k_lim=50,FWHM_change=True,kolb_factor=False,method="analytical",scale=0.01,beta=2.5):     
        """
        Calculate the loaded waves' transmision using calculated shifts
        Can be done using an analytical gaussian method, or a numerical gaussian/moffat method
        Fibre is currently modelled as a circular aperture

        INPUTS:
        k_lim: float, default=50
            number of terms to compute the sum to for the analytic transmission solution, 50 is a safe value
        FWHM_change: string, True or False
            whether to change the monochromatic FWHM with airmass and wavelength
        kolb_factor: boolean, True or False
            whether to use the kolb factor in FWHM change, as per https://www.eso.org/observing/etc/doc/helpfors.html
        method: string, "analytical", "numerical guassian", "numerical moffat"
            which method to use for calculating transmission
        scale: float, default=0.01
            scale to use in the numerical methods, arcsec/pixel
        beta: float, default=2.5
            moffat index to use

        OUTPUTS:
        Input dictionary:
        self.FWHM_change, self.kolb_factor, self.k_lim, self.method: boolean, boolean, float, string
            used for plotting data later
        
        Output dictionary:
        self.wave_transmissions
            transmissions of the wave calculated through the chosen method. Form is [[airmass 1 transmissions...][airmass 2 transmissions...][...]...]
        """
        self.input_parameters['FWHM_change']=FWHM_change #Store all these parameters for plotting later
        self.input_parameters['kolb_factor']=kolb_factor
        self.input_parameters['k_lim']=k_lim
        self.input_parameters['method']=method
        self.input_parameters['beta']=beta
        self.input_parameters['scale']=scale

        wave_wavelengths = self.output_parameters['wave_wavelengths']
        airmasses=self.output_parameters['airmasses']
        shifts=self.output_parameters['shifts']
        fibre_diameter=self.output_parameters['fibre_diameter']

        median_FWHM=self.input_parameters['median_FWHM']
        median_FWHM_lambda=self.input_parameters['median_FWHM_lambda']

        if FWHM_change == True: #Is there a dependence of the FWHM on airmass and wavelength?
            FWHMs = [] #List of the FWHM across the wave for each airmass, [[airmass 1 wave FWHM],[airmass 2 wave FWHM],....]
            for i in airmasses: #For each airmass, calculate the FWHM for each wavelength in the wave
                FWHMs.append(trans_calc.calculate_FWHM(wave_wavelengths,i,median_FWHM,median_FWHM_lambda,kolb_factor))
            FWHMs=np.array(FWHMs) * u.arcsec
            if method=="analytical": #use analytical method
                wave_transmissions=trans_calc.analytical_gaussian(fibre_diameter,FWHMs,shifts,k_lim) #Transmission of light into fibre using changing FWHM       
            elif method=="numerical gaussian": #Use numerical gaussian method. Function does not take in arrays, so a loop is needed
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_gaussian(fibre_diameter,FWHMs[i][o],shifts[i][o],scale) #Transmission for given wavelength at given airmass
                        trans_list.append(trans)
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
            elif method=="numerical moffat": #Use numerical moffat method. Function does not take in arrays, so a loop is needed
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_moffat(fibre_diameter,FWHMs[i][o],shifts[i][o],scale,beta=beta) #Transmission for given wavelength at given airmass
                        trans_list.append(trans) 
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
        else:
            FWHMs=np.full((len(airmasses),len(wave_wavelengths)),median_FWHM) * u.arcsec
            if method=="analytical":
                wave_transmissions=trans_calc.analytical_gaussian(fibre_diameter,median_FWHM,shifts,k_lim) #Transmission of light into fibre using changing FWHM
            elif method=="numerical gaussian":
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_gaussian(fibre_diameter,FWHMs[i][o],shifts[i][o],scale) #Transmission for given wavelength at given airmass
                        trans_list.append(trans)
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list
            elif method=="numerical moffat":
                wave_transmissions=[]
                for i in range(0,len(airmasses)): #For every airmass, need transmissions
                    trans_list=[] #List of transmissions for the airmass
                    for o in range(0,len(wave_wavelengths)): #For every monochromatic PSF, need it's transmission in that airmass
                        trans=trans_calc.numerical_moffat(fibre_diameter,FWHMs[i][o],shifts[i][o],scale,beta=beta) #Transmission for given wavelength at given airmass
                        trans_list.append(trans) 
                    wave_transmissions.append(trans_list) #Append this airmass transmissions' to the list

        self.output_parameters['wave_transmissions']=wave_transmissions
        self.output_parameters['FWHMs']=FWHMs
        
    #Below is not documented
    def plot_snaptransmissions(self, normalise="none"):
        """
        Plots the wavelength vs transmission graph 
        Comes after load_wave, load_airmasses, calculate_snapshifts, calculate_snaptransmissions

        INPUTS:
        normalise: string, "none", "zenith", "centre", "both"
            what to normalise the transmission to; either no normalisation, relative to zenith, relative to transmission or fibre centre wavelength,
            or both the latter

        OUTPUTS:
        Graphs:
            corresponding graph
        """
        airmasses=self.output_parameters['airmasses']
        ZA_range=self.input_parameters['ZA_range']
        HA_range=self.input_parameters['HA_range']
        
        wave_wavelengths=self.output_parameters['wave_wavelengths']
        fibre_diameter=self.output_parameters['fibre_diameter']
        wave_transmissions=self.output_parameters['wave_transmissions']


        fibrecentre_waveref=self.input_parameters['fibrecentre_waveref']
        guide_waveref=self.input_parameters['guide_waveref']
        median_FWHM=self.input_parameters['median_FWHM']
        median_FWHM_lambda=self.input_parameters['median_FWHM_lambda']
        kolb_factor=self.input_parameters['kolb_factor']
        regime=self.input_parameters['regime']
        res=self.input_parameters['res']
        k_lim=self.input_parameters['k_lim']
        FWHM_change=self.input_parameters['FWHM_change']
        scale=self.input_parameters['scale']
        method=self.input_parameters['method']
        beta=self.input_parameters['beta']
        band=self.input_parameters['band']
        reposition=self.input_parameters['reposition']
        targ_dec=self.input_parameters['targ_dec']


        weights = np.arange(1, len(airmasses)+1)
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)

        if ZA_range !=[]: #Zenith angle plots
            fig, ax = plt.subplots(figsize=(10,7))
            plt.axvline(fibrecentre_waveref.value*1000,color='black',linewidth=0.5,label='Fibre Centre Wavelength')

            if normalise == "zenith":
                print("Normalised to Zenith Transmission")
                if FWHM_change==True:
                    zenith_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,1,median_FWHM,median_FWHM_lambda,kolb_factor)
                else:
                    zenith_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                zenith_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,1,guide_waveref,self.conditions)
                zenith_shifts=Atmospheric_diffraction(wave_wavelengths,1,guide_waveref,self.conditions)-zenith_centre_shift
                if method == "analytical":
                    zenith_transmission = trans_calc.analytical_gaussian(fibre_diameter,zenith_FWHMs,zenith_shifts,k_lim)
                if method == "numerical gaussian":
                    zenith_transmission = []
                    for i in range(0,len(zenith_FWHMs)):
                        zenith_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale))
                if method == "numerical moffat":
                    zenith_transmission = []
                    for i in range(0,len(zenith_FWHMs)):
                        zenith_transmission.append(trans_calc.numerical_moffat(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale,beta=beta))
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(zenith_transmission),label='Zenith Angle = %2.0f' %(ZA_range[i]),color=cmap.to_rgba(i+1))    
                plt.ylabel("Transmission Relative to Zenith")

            if normalise == "centre":
                print("Normalised to Reference Wavelength Transmission")
                peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]          
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(wave_transmissions[i][peak]),label='Zenith Angle = %2.0f' %(ZA_range[i]),color=cmap.to_rgba(i+1))    
                plt.ylabel("Transmission Relative to Reference Wavelength")

            if normalise =="both":
                print("Normalised to Both Zenith Transmission and Reference Wavelength Transmission")
                if FWHM_change==True:
                    zenith_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,1,median_FWHM,median_FWHM_lambda,kolb_factor)
                else:
                    zenith_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                zenith_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,1,guide_waveref,self.conditions)
                zenith_shifts=Atmospheric_diffraction(wave_wavelengths,1,guide_waveref,self.conditions)-zenith_centre_shift
                if method == "analytical":
                    zenith_transmission = trans_calc.analytical_gaussian(fibre_diameter,zenith_FWHMs,zenith_shifts,k_lim)
                if method == "numerical gaussian":
                    zenith_transmission = []
                    for i in range(0,len(zenith_FWHMs)):
                        zenith_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale))
                if method == "numerical moffat":
                    zenith_transmission = []
                    for i in range(0,len(zenith_FWHMs)):
                        zenith_transmission.append(trans_calc.numerical_moffat(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale,beta=beta))
                peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(zenith_transmission)/(np.array(wave_transmissions)[i][peak]/np.array(zenith_transmission)[peak]),label='Zenith Angle = %2.0f' %(self.ZA_range[i]),color=cmap.to_rgba(i+1))
                    plt.ylabel("Transmission (Normalised to Zenith and Ref. Wave. Transmission)")

            if normalise == "none":
                print("No Normalisation, Raw Transmission")
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i]),label='Zenith Angle = %2.0f' %(ZA_range[i]),color=cmap.to_rgba(i+1))
                plt.ylabel("Transmission")

            plt.xlabel("Wavelength [nm]")
            plt.title('Fibre = %s, Guide = %s, %s %s, FWHM Change = %s, Repos = %s, Method = %s' %(fibrecentre_waveref,guide_waveref,regime,band,FWHM_change,reposition,method))
            plt.ylim(0,1.3)
            plt.legend()

        if HA_range != []: #HA plots
            fig, ax = plt.subplots(figsize=(10,7))
            plt.axvline(fibrecentre_waveref.value*1000,color='black',linewidth=0.5,label='Fibre Centre Wavelength')
            meridian_airmass=self.output_parameters['meridian_airmass']
            if normalise == "meridian":
                print("Normalised to Target's Meridian Transmission (LHA=0h)")
                if FWHM_change==True:
                    merid_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,meridian_airmass,median_FWHM,median_FWHM_lambda,kolb_factor)
                else:
                    merid_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                merid_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,meridian_airmass,guide_waveref,self.conditions)
                merid_shifts=Atmospheric_diffraction(wave_wavelengths,meridian_airmass,guide_waveref,self.conditions)-merid_centre_shift
                if method == "analytical":
                    merid_transmission = trans_calc.analytical_gaussian(fibre_diameter,merid_FWHMs,merid_shifts,k_lim)
                if method == "numerical gaussian":
                    merid_transmission = []
                    for i in range(0,len(merid_FWHMs)):
                        merid_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale))
                if method == "numerical moffat":
                    merid_transmission = []
                    for i in range(0,len(merid_FWHMs)):
                        merid_transmission.append(trans_calc.numerical_moffat(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale,beta=beta))
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(merid_transmission),label='HA = %2.2fh' %(HA_range[i]),color=cmap.to_rgba(i+1))
                plt.ylabel("Transmission Relative to Target's at Meridian")

            if normalise == "centre":
                print("Normalised to Reference Wavelength Transmission")
                peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(wave_transmissions[i][peak]),label='HA = %2.2fh' %(HA_range[i]),color=cmap.to_rgba(i+1))
                plt.ylabel("Transmission Relative to Reference Wavelength")

            if normalise =="both":
                print("Normalised to Both Target's Meridian Transmission and Reference Wavelength Transmission")
                if FWHM_change==True:
                    merid_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,meridian_airmass,median_FWHM,median_FWHM_lambda,kolb_factor)
                else:
                    merid_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                merid_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,meridian_airmass,guide_waveref,self.conditions)
                merid_shifts=Atmospheric_diffraction(wave_wavelengths,meridian_airmass,guide_waveref,self.conditions)-merid_centre_shift
                if method == "analytical":
                    merid_transmission = trans_calc.analytical_gaussian(fibre_diameter,merid_FWHMs,merid_shifts,k_lim)
                if method == "numerical gaussian":
                    merid_transmission = []
                    for i in range(0,len(merid_FWHMs)):
                        merid_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale))
                if method == "numerical moffat":
                    merid_transmission = []
                    for i in range(0,len(merid_FWHMs)):
                        merid_transmission.append(trans_calc.numerical_moffat(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale,beta=beta))
                peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(merid_transmission)/(np.array(wave_transmissions)[i][peak]/np.array(merid_transmission)[peak]),label='HA = %2.0fh' %(HA_range[i]),color=cmap.to_rgba(i+1))
                    plt.ylabel("Transmission (Normalised to Target at Meridian and Ref. Wave. Transmission)")

            if normalise == "none":
                for i in range(0,len(wave_transmissions)):
                    plt.plot(wave_wavelengths,np.array(wave_transmissions[i]),label='HA = %2.2fh' %(HA_range[i]),color=cmap.to_rgba(i+1))
                plt.ylabel("Transmission")

            plt.xlabel("Wavelength [nm]")
            plt.title('Fibre = %s, Guide = %s, %s %s, FWHM Change = %s, Dec = %2.2f, Repos = %s, Method = %s' %(fibrecentre_waveref,guide_waveref,regime,band,FWHM_change,targ_dec.value,reposition,method))
            plt.ylim(0,1.3)
            plt.legend()
        
    def calculate_integtransmissions(self, start_HA, end_HA, repos_interval, intervals, method="analytical"):
        """
        Calculates and plots the wavelength vs transmission graph for various HA with a given reposition interval 
        Comes after load_wave

        INPUTS:
        start_HA, end_HA: float
            start and end hour angles for the plot
        repos_interval: float
            reposition the fibre after this interval
        intervals: float
            number of samples to take between each fibre position

        OUTPUTS:
        Graphs:
            corresponding graph
        """
        HA_ranges=[]
        for i in range(0,int((end_HA-start_HA)/repos_interval)):
            HA_ranges.append(np.linspace(i*repos_interval+start_HA,(i+1)*repos_interval+start_HA,intervals))
        
        transmissions=[]

        for i in range(0,int((end_HA-start_HA)/repos_interval)):
            self.load_airmasses(HA_range=HA_ranges[i],targ_dec=40 * u.deg)
            self.calculate_snapshifts(fibrecentre_waveref = 1 * u.micron,plot=False, reposition=False, guide_waveref=0.6 * u.micron)
            self.calculate_snaptransmissions(FWHM_change=True,method=method)
            transmissions.append(self.output_parameters['wave_transmissions'])

        weights = np.arange(0, int(intervals*(end_HA-start_HA)/repos_interval)+1)
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        style=['-','--','-.']

        fig, ax = plt.subplots(figsize=(10,7))
        for i in range(0,len(transmissions)):
            for o in range(0,len(transmissions[i])):
                factor=0
                if o == 0: 
                    factor = -1
            
                plt.plot(self.output_parameters['wave_wavelengths'],transmissions[i][o],linestyle=style[o],color=cmap.to_rgba(o+i*len(transmissions[0])+factor),label="Repos = %2.2fh, HA = %2.2fh" %(i,HA_ranges[i][o]))
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmission")
        plt.ylim(0,1.1)

    def seperation_vs_zenith(self,res,regime,min_band,max_band):
        """
        Calculates the zenith angle vs red/blue PSF seperation values
        Comes after load_airmasses

        INPUTS:
        res: string
            what to normalise the transmission to; either no normalisation, relative to zenith, relative to transmission of fibre centre wavelength,
            or both the latter
        regime: string
        
        min_band: string

        max_band: string

        OUTPUTS:
        seperation: float
            seperation values for the red/blue PSF for the given res/regime/band over the loaded airmasses
        """
        Config_regime = ConfigObj('./Architecture_parameters/'+regime+'_channel_conf.ini') #Loads VIS or NIR parameters

        sampling = -int(Config_regime[res]['Bands'][min_band]['wave_min'])+int(Config_regime[res]['Bands'][max_band]['wave_max'])

        self.load_wave(res,regime,min_band,max_band,sampling * u.nm)
    
        self.calculate_snapshifts(fibrecentre_waveref = 1 * u.micron,plot=False, reposition=True, guide_waveref=0.6 * u.micron)
        seperation=[]
        for i in self.output_parameters['shifts']:
            seperation.append(abs(i[0]-i[1]).value)

        # fig, ax = plt.subplots(figsize=(10,7))
        # plt.axhline(self.output_parameters['fibre_diameter'].value,color='black',linewidth=0.5,label='%s Fibre Diameter' %(regime))
        # plt.plot(self.input_parameters['ZA_range'],seperation,label="%s %s" %(regime,self.input_parameters['band']))
        # plt.ylabel("Seperation (arcsecs)")
        # plt.xlabel("Zenith Angle")
        # plt.legend()

        return seperation

    def compare_moffat_gaussian(self,normalise='none'):
        airmasses=self.output_parameters['airmasses']
        ZA_range=self.input_parameters['ZA_range']
        HA_range=self.input_parameters['HA_range']
        fibrecentre_waveref=self.input_parameters['fibrecentre_waveref']
        methods=["gaussian", "moffat"]
        style=["-","-."]
        fig, ax = plt.subplots(figsize=(10,7))
        plt.axvline(fibrecentre_waveref.value*1000,color='black',linewidth=0.5,label='Fibre Centre Wavelength')

        for o in range(0,2):
            self.calculate_snaptransmissions(k_lim=30, FWHM_change=True, kolb_factor=True, method="numerical "+methods[o],scale=0.01)
            
            wave_wavelengths=self.output_parameters['wave_wavelengths']
            fibre_diameter=self.output_parameters['fibre_diameter']
            wave_transmissions=self.output_parameters['wave_transmissions']

            
            guide_waveref=self.input_parameters['guide_waveref']
            median_FWHM=self.input_parameters['median_FWHM']
            median_FWHM_lambda=self.input_parameters['median_FWHM_lambda']
            kolb_factor=self.input_parameters['kolb_factor']
            regime=self.input_parameters['regime']
            res=self.input_parameters['res']
            k_lim=self.input_parameters['k_lim']
            FWHM_change=self.input_parameters['FWHM_change']
            scale=self.input_parameters['scale']
            method=self.input_parameters['method']
            beta=self.input_parameters['beta']
            band=self.input_parameters['band']
            reposition=self.input_parameters['reposition']
            targ_dec=self.input_parameters['targ_dec']

            weights = np.arange(1, len(airmasses)+1)
            norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)

            if ZA_range !=[]: #Zenith angle plots
                


                if normalise == "zenith":
                    print("Normalised to Zenith Transmission")
                    if FWHM_change==True:
                        zenith_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,1,median_FWHM,median_FWHM_lambda,kolb_factor)
                    else:
                        zenith_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                    zenith_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,1,guide_waveref,self.conditions)
                    zenith_shifts=Atmospheric_diffraction(wave_wavelengths,1,guide_waveref,self.conditions)-zenith_centre_shift
                    if method == "analytical":
                        zenith_transmission = trans_calc.analytical_gaussian(fibre_diameter,zenith_FWHMs,zenith_shifts,k_lim)
                    if method == "numerical gaussian":
                        zenith_transmission = []
                        for i in range(0,len(zenith_FWHMs)):
                            zenith_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale))
                    if method == "numerical moffat":
                        zenith_transmission = []
                        for i in range(0,len(zenith_FWHMs)):
                            zenith_transmission.append(trans_calc.numerical_moffat(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale,beta=beta))
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(zenith_transmission),'ZA = %2.0f %s' %(ZA_range[i],method[o]),color=cmap.to_rgba(i+1),linestyle=style[o])    
                    plt.ylabel("Transmission Relative to Zenith")

                if normalise == "centre":
                    print("Normalised to Reference Wavelength Transmission")
                    peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]          
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(wave_transmissions[i][peak]),'ZA = %2.0f %s' %(ZA_range[i],methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])    
                    plt.ylabel("Transmission Relative to Reference Wavelength")

                if normalise =="both":
                    print("Normalised to Both Zenith Transmission and Reference Wavelength Transmission")
                    if FWHM_change==True:
                        zenith_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,1,median_FWHM,median_FWHM_lambda,kolb_factor)
                    else:
                        zenith_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                    zenith_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,1,guide_waveref,self.conditions)
                    zenith_shifts=Atmospheric_diffraction(wave_wavelengths,1,guide_waveref,self.conditions)-zenith_centre_shift
                    if method == "analytical":
                        zenith_transmission = trans_calc.analytical_gaussian(fibre_diameter,zenith_FWHMs,zenith_shifts,k_lim)
                    if method == "numerical gaussian":
                        zenith_transmission = []
                        for i in range(0,len(zenith_FWHMs)):
                            zenith_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale))
                    if method == "numerical moffat":
                        zenith_transmission = []
                        for i in range(0,len(zenith_FWHMs)):
                            zenith_transmission.append(trans_calc.numerical_moffat(fibre_diameter,zenith_FWHMs[i],zenith_shifts[i],scale,beta=beta))
                    peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(zenith_transmission)/(np.array(wave_transmissions)[i][peak]/np.array(zenith_transmission)[peak]),'ZA = %2.0f %s' %(ZA_range[i],methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])
                        plt.ylabel("Transmission (Normalised to Zenith and Ref. Wave. Transmission)")

                if normalise == "none":
                    print("No Normalisation, Raw Transmission")
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i]),label='ZA = %2.0f %s' %(ZA_range[i],methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])
                    plt.ylabel("Transmission")

                plt.xlabel("Wavelength [nm]")
                plt.title('Fibre = %s, Guide = %s, %s %s, FWHM Change = %s, Repos = %s' %(fibrecentre_waveref,guide_waveref,regime,band,FWHM_change,reposition))
                plt.ylim(0,1.3)
                plt.legend()

            if HA_range != []: #HA plots
                meridian_airmass=self.output_parameters['meridian_airmass']
                if normalise == "meridian":
                    print("Normalised to Target's Meridian Transmission (LHA=0h)")
                    if FWHM_change==True:
                        merid_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,meridian_airmass,median_FWHM,median_FWHM_lambda,kolb_factor)
                    else:
                        merid_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                    merid_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,meridian_airmass,guide_waveref,self.conditions)
                    merid_shifts=Atmospheric_diffraction(wave_wavelengths,meridian_airmass,guide_waveref,self.conditions)-merid_centre_shift
                    if method == "analytical":
                        merid_transmission = trans_calc.analytical_gaussian(fibre_diameter,merid_FWHMs,merid_shifts,k_lim)
                    if method == "numerical gaussian":
                        merid_transmission = []
                        for i in range(0,len(merid_FWHMs)):
                            merid_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale))
                    if method == "numerical moffat":
                        merid_transmission = []
                        for i in range(0,len(merid_FWHMs)):
                            merid_transmission.append(trans_calc.numerical_moffat(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale,beta=beta))
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(merid_transmission),label='HA = %2.2fh %s' %(HA_range[i], methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])
                    plt.ylabel("Transmission Relative to Target's at Meridian")

                if normalise == "centre":
                    print("Normalised to Reference Wavelength Transmission")
                    peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(wave_transmissions[i][peak]),label='HA = %2.2fh %s' %(HA_range[i],methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])
                    plt.ylabel("Transmission Relative to Reference Wavelength")

                if normalise =="both":
                    print("Normalised to Both Target's Meridian Transmission and Reference Wavelength Transmission")
                    if FWHM_change==True:
                        merid_FWHMs=trans_calc.calculate_FWHM(wave_wavelengths,meridian_airmass,median_FWHM,median_FWHM_lambda,kolb_factor)
                    else:
                        merid_FWHMs=np.full(len(wave_wavelengths),median_FWHM) * u.arcsec
                    merid_centre_shift=Atmospheric_diffraction(fibrecentre_waveref,meridian_airmass,guide_waveref,self.conditions)
                    merid_shifts=Atmospheric_diffraction(wave_wavelengths,meridian_airmass,guide_waveref,self.conditions)-merid_centre_shift
                    if method == "analytical":
                        merid_transmission = trans_calc.analytical_gaussian(fibre_diameter,merid_FWHMs,merid_shifts,k_lim)
                    if method == "numerical gaussian":
                        merid_transmission = []
                        for i in range(0,len(merid_FWHMs)):
                            merid_transmission.append(trans_calc.numerical_gaussian(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale))
                    if method == "numerical moffat":
                        merid_transmission = []
                        for i in range(0,len(merid_FWHMs)):
                            merid_transmission.append(trans_calc.numerical_moffat(fibre_diameter,merid_FWHMs[i],merid_shifts[i],scale,beta=beta))
                    peak = np.where(wave_wavelengths.value==fibrecentre_waveref.value*1000)[0][0]
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i])/np.array(merid_transmission)/(np.array(wave_transmissions)[i][peak]/np.array(merid_transmission)[peak]),label='HA = %2.0fh %s' %(HA_range[i],methods[o]),color=cmap.to_rgba(i+1),linestyle=style[o])
                        plt.ylabel("Transmission (Normalised to Target at Meridian and Ref. Wave. Transmission)")

                if normalise == "none":
                    for i in range(0,len(wave_transmissions)):
                        plt.plot(wave_wavelengths,np.array(wave_transmissions[i]),label='HA = %2.2fh' %(HA_range[i]),color=cmap.to_rgba(i+1),linestyle=style[o])
                    plt.ylabel("Transmission")

                plt.xlabel("Wavelength [nm]")
                plt.title('Fibre = %s, Guide = %s, %s %s, FWHM Change = %s, Dec = %2.2f, Repos = %s' %(fibrecentre_waveref,guide_waveref,regime,band,FWHM_change,targ_dec.value,reposition))
                plt.ylim(0,1.3)
                plt.legend()