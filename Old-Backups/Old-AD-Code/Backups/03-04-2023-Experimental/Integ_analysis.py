from AD_analysis import *

def observation_transmission(wave_wavelengths,wave_transmissions):
    integ_transmission=[]
    for i in range(0,len(wave_wavelengths)):
        trans=0
        for o in range(0,len(wave_transmissions)):
            trans=trans+wave_transmissions[o][i]
        trans_mean=trans/len(wave_transmissions)
        integ_transmission.append(trans_mean)
    return integ_transmission

def integ_trans(analysis,aperturecentre_waverefs,guide,parallatic): 
    integ_transmissions=[]
    
    for aperture_val in aperturecentre_waverefs:
        analysis.calculate_shifts(aperturecentre_waveref = aperture_val,reposition=False, guide_waveref=guide,parallatic=parallatic)
        analysis.calculate_transmissions()
        integ_transmission=observation_transmission(analysis.output_parameters['wave_wavelengths'],analysis.output_parameters['wave_transmissions'])
        integ_transmissions.append(integ_transmission)
    
    old_shifts=analysis.output_parameters['shifts'].copy()
    for i in range(0,len(analysis.output_parameters['shifts'])):
        for o in range(0,len(analysis.output_parameters['shifts'][i])):
            analysis.output_parameters['shifts'][i][o]=0
    analysis.calculate_transmissions()
    opt_transmission=observation_transmission(analysis.output_parameters['wave_wavelengths'],analysis.output_parameters['wave_transmissions'])
    
    analysis.output_parameters['shifts']=old_shifts
    
    return integ_transmissions,opt_transmission
    

def integ_metric(normalised_transmissions,metric):
    if metric == "min trans":
        return min(normalised_transmissions)
    
    else:
        print("Metric doesnt exist")
        return


