from AD_analysis import *
import matplotlib as mpl

def observation_transmission(transmissions):
    """ 
    Takes multiple transmissions from the analysis and takes the average transmissions for each wavelength
    This is used for average trans over an observation/transmission
    
    INPUTS:
    transmissions : lists of lists of floats
        output transmissions from the analysis code. This is in the form of [[Airmass 1 trans..][Airmass 2 trans..]..]
        
    OUTPUTS:
    integ_transmissions: list of floats
        average transmissions for each wavelength.
    """
    integ_transmission=[]
    for i in range(0,len(transmissions[0])):
        trans=0
        for o in range(0,len(transmissions)):
            trans=trans+transmissions[o][i]
        trans_mean=trans/len(transmissions)
        integ_transmission.append(trans_mean)
    return integ_transmission

def integ_trans(analysis,aperturecentre_waverefs,guide,parallatic,centre_index=0): 
    
    integ_transmissions=[]
    for aperture_val in aperturecentre_waverefs:
        analysis.calculate_shifts(guide, aperture_val, centring_index=centre_index,reposition = False, parallatic=parallatic)
        analysis.calculate_transmissions()
        integ_transmission=observation_transmission(analysis.output['transmissions'])
        integ_transmissions.append(integ_transmission)
    
    old_shifts=analysis.output['shifts'].copy()
    for i in range(0,len(analysis.output['shifts'])):
        for o in range(0,len(analysis.output['shifts'][i])):
            analysis.output['shifts'][i][o]=0
    analysis.calculate_transmissions()
    opt_transmission=observation_transmission(analysis.output['transmissions'])
    
    analysis.output['shifts']=old_shifts
    
    return integ_transmissions,opt_transmission
    
def integ_metric(normalised_transmissions,metric):
    if metric == "min trans":
        return min(normalised_transmissions)
    
    else:
        print("Metric doesnt exist")
        return
    
def track_plot(analysis,y_axis):
    if y_axis == "centring":
        HA_range=analysis.input['HA_range']
        aperture=analysis.input['aperture_waveref']
        guide=analysis.input['guide_waveref']
        targ_dec=analysis.input['targ_dec']
        centring=analysis.input['centring_index']

        d = analysis.output['shifts']
        s=analysis.output['shifts_no_para']
        c = analysis.output['centre_shift']
        
        xs=[]
        ys=[]
        for count,q in enumerate(analysis.output['delta_para_angles']):
            x=(s[count]+c)*np.sin(q)
            y=(s[count]+c)*np.cos(q)-c
            xs.append(x)
            ys.append(y)
        xs_new=[]
        ys_new=[]
        for i in range(0,len(xs[0])):
            x_new=[]
            y_new=[]
            for o in range(0,len(xs)):
                x_new.append(xs[o][i].value)
                y_new.append(ys[o][i].value)
                if round(abs(np.sqrt(xs[o][i].value**2+ys[o][i].value**2)),5) != round(abs(d[o][i]).value,5):
                    print("plot NOT EQUAL to code output - there is an error")
            xs_new.append(x_new)
            ys_new.append(y_new)
            
        weights = np.arange(0, len(analysis.output['wavelengths']))
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
        fig, ax = plt.subplots(figsize=[6,6]) 
        circle1 = plt.Circle((0, 0), analysis.output['aperture_diameter'].value/2, color='black', fill=False, label='Aperture')
        ax.add_patch(circle1)    
        plt.axvline(0,color='black',linestyle='--',linewidth=0.7,label="HA = {}h".format(analysis.input['HA_range'][analysis.input['centring_index']]))
        for i in range(0,len(xs_new)):
            plt.plot(xs_new[i],np.array(ys_new[i]),marker='x',color=cmap.to_rgba(i),label="%2.0f nm" %(round(analysis.output['wavelengths'][i].value,0)))
        plt.scatter(0,-c,label='Guide = {}nm'.format(round(analysis.input['guide_waveref'].value*1000)),color='black',marker='+')
        plt.title("HA: {}-{}h, Dec = {}, Guide = {}, Aperture = {} at {}h".format(HA_range[0],HA_range[-1],analysis.input['targ_dec'],guide,aperture,HA_range[centring]))
        plt.ylim(-0.5,0.5)
        plt.xlim(-0.5,0.5)
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        plt.legend()
        

    
    if y_axis == "PA":
        HA_range=analysis.input['HA_range']
        aperture=analysis.input['aperture_waveref']
        guide=analysis.input['guide_waveref']
        targ_dec=analysis.input['targ_dec']
        centring=analysis.input['centring_index']

        d = analysis.output['shifts']
        s=analysis.output['shifts_no_para']
        c = analysis.output['centre_shift']

        xs=[]
        ys=[]
        for count,q in enumerate(analysis.output['raw_para_angles']):
            if targ_dec.value > analysis.conditions['latitude'].value:
                x=(s[count]+c)*np.sin(q)-c*np.sin(analysis.output['raw_para_angles'][centring])
                y=(s[count]+c)*np.cos(q)-c*np.cos(analysis.output['raw_para_angles'][centring])
                xs.append(x)
                ys.append(y)
            elif targ_dec.value < analysis.conditions['latitude'].value:
                x=(s[count]+c)*np.sin(q)-c*np.sin(analysis.output['raw_para_angles'][centring])
                y=(s[count]+c)*np.cos(q)-c*np.cos(analysis.output['raw_para_angles'][centring])
                xs.append(x)
                ys.append(y)
        xs_new=[]
        ys_new=[]
        for i in range(0,len(xs[0])):
            x_new=[]
            y_new=[]
            for o in range(0,len(xs)):
                x_new.append(xs[o][i].value)
                y_new.append(ys[o][i].value)
                
                if round(abs(np.sqrt(xs[o][i].value**2+ys[o][i].value**2)),5) != round(abs(d[o][i]).value,5):
                    print("plot NOT EQUAL to code output - there is an error")
            xs_new.append(x_new)
            ys_new.append(y_new)
            
        weights = np.arange(0, len(analysis.output['wavelengths']))
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap='seismic')
        fig, ax = plt.subplots(figsize=[6,6]) 
        circle1 = plt.Circle((0, 0), analysis.output['aperture_diameter'].value/2, color='black', fill=False, label='Aperture')
        ax.add_patch(circle1)    
        plt.axvline(0,color='black',linestyle='--',linewidth=0.7,label="PA = 0")
        for i in range(0,len(xs_new)):
            plt.plot(xs_new[i],np.array(ys_new[i]),marker='x',color=cmap.to_rgba(i),label="%2.0f nm" %(round(analysis.output['wavelengths'][i].value,0)))
        plt.ylim(-0.5,0.5)
        plt.xlim(-0.5,0.5)
        plt.scatter(-c*np.sin(analysis.output['raw_para_angles'][centring]),-c*np.cos(analysis.output['raw_para_angles'][centring]),label='Guide = {}nm'.format(round(analysis.input['guide_waveref'].value*1000)),color='black',marker='+')
        plt.legend()
        plt.title("HA: {}-{}h, Dec = {}, Guide = {}, Aperture = {} at {}h".format(HA_range[0],HA_range[-1],analysis.input['targ_dec'],guide,aperture,HA_range[centring]))
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")


