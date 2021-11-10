"""
Script to make ED Figure 4 from Nimmo et al. 2021b
ACF and PS of burst B3 at 31.25ns

Kenzie Nimmo 2021

Data you need:
- SFXC 31.25ns filterbank containing burst B3
    - pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil
- power spectrum of the 31.25ns time profile (and downsampled version)
    - PS_B3_IF3568_31.25ns.npy (PS_B3_IF3568_31.25ns.f8.npy)
- power law and power law + lorentzian fits to the power spectrum determined using STINGRAY
    - PL_fit_B3.npy, PL_lor_fit_B3.npy


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from load_file import load_filterbank
from ACF_funcs import crosscoef, autocorr,lorentz
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib as mpl

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'

def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average                                                                                                                                                 
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



data,extent,tsamp = load_filterbank('./data_products/pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil')

#compute the ACF per bright subband and average together
ACF=np.zeros(3000)
ACFoff=np.zeros(3000)
for i in [3,5,6,8]: #brightest subbands
    burst_chan = data[i-1,25300000:25305000]
    boff=data[i-1,10000:10000+len(burst_chan)]
    burst_chan-=np.mean(boff)
    boff-=np.mean(boff)
    burst_chan/=np.std(boff)
    boff/=np.std(boff)
    ACF += autocorr(burst_chan[1000:4000],len(burst_chan[1000:4000]))
    ACFoff +=autocorr(boff[1000:4000],len(boff[1000:4000]))

#correct for adding four subbands together
ACF/=4.
ACFoff/=4.

#make bins in microseconds
ACFbins=np.arange(1,len(ACF)+1,1)*31.25e-9*1e6 #microseconds

#fitting the different timescales with Lorentzian functions
popt,pcov=curve_fit(lorentz,ACFbins[80:1500],(ACF-ACFoff)[80:1500],sigma=ACFoff[80:1500])
popt2,pcov2=curve_fit(lorentz,ACFbins[5:200],(ACF-ACFoff)[5:200],sigma=ACFoff[5:200])
popt3,pcov3=curve_fit(lorentz,ACFbins[0:20],(ACF-ACFoff)[0:20],sigma=ACFoff[0:20],p0=[0.06125,0.1,0.1],maxfev=2000)

#used to measure the timescales with fit uncertainties
"""
chisq = np.sum((lorentz(ACFbins[:20],*popt3)-ACF[0:20])**2/(np.std(ACFoff[20:30])**2))
print("chisq", chisq)
predicted_chisq = len(ACF[0:20])-3 #degrees of freedom                                                                                                                               
print("dof", predicted_chisq)
pval = 1 - stats.chi2.cdf(chisq, predicted_chisq)
pval2 = stats.chi2.sf(chisq,predicted_chisq)
print("pval", pval)
#print " pval2", pval2                                                                                                                                                           

correct_pcov = pcov*(predicted_chisq/chisq)
print("Characteristic scale", popt3[0], "+-",correct_pcov[0][0],"MHz")
"""

#mirror the ACFs
ACF = np.concatenate((ACF[::-1],ACF))
ACFoff = np.concatenate((ACFoff[::-1],ACFoff))
ACFbins = np.concatenate((ACFbins[::-1],-ACFbins))


#plot
fig = plt.figure(figsize=(7,7))
gs = gridspec.GridSpec(ncols=1, nrows=2, left=0.09,right=0.45, wspace=0.0, hspace=0.0,height_ratios=[1,4]) 
gs2 = gridspec.GridSpec(ncols=1, nrows=2, left=0.56,right=0.98,bottom=0.22,top=0.46, wspace=0.0, hspace=0.0)      
gs4 = gridspec.GridSpec(ncols=1, nrows=1, left=0.56,right=0.98,bottom=0.06,top=0.18, wspace=0.0, hspace=0.0)
gs3 = gridspec.GridSpec(ncols=1, nrows=3, left=0.56,right=0.98,bottom=0.53,top=0.96, wspace=0.0, hspace=0.0,height_ratios=[2,1,1])

ax1 = fig.add_subplot(gs[0,0]) #summed profile
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('B3',u'31.25ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

av_prof = np.mean(data[0:11,25300000:25305000],axis=0)
av_off = np.mean(data[0:11,10000:10000+len(av_prof)],axis=0)
av_prof-=np.mean(av_off)
av_off-=np.mean(av_off)
av_prof/=np.std(av_prof)

time = np.arange(0.,float(len(av_prof)),1.)
time-=1000.
time*=31.25e-9*1e6
ax1.set_ylabel('S/N')
ax1.set_xlim(0,(3000)*31.25e-9*1e6)
ax1.plot(time,av_prof,color='k',linestyle='steps-mid',lw=0.5)
plt.setp(ax1.get_xticklabels(), visible=False)
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # dynamic spectrum of profiles     
for f in range(11):
    prof = data[f,25300000:25305000]
    profoff = data[f,10000:10000+len(prof)]
    prof-=np.mean(profoff)
    profoff-=np.mean(profoff)
    prof/=np.std(profoff)
    ax2.plot(time,(prof*(16.0/44.3313523288)+1256+(f*16)),zorder=data.shape[0]-f,color='k',lw=0.5,linestyle='steps-mid') # scale the height so there is no overlap
    

ax2.set_xlim(0,(3000)*31.25e-9*1e6)
ax2.set_xlabel(u'Time [\u03bcs]')
ax2.set_ylabel('Frequency [MHz]')
point = ax2.scatter(0,1300, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs2[0,0])
point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax3.plot(ACFbins,ACFoff,lw=0.7,color='#756D76', label='Average off burst ACF')
ax3.plot(ACFbins,ACF,lw=0.7,color='orange', label='Average on burst ACF')  
ax3.set_xlim(-50,50)
ax3.set_ylabel('ACF')
ax3.set_ylim(-0.1,0.18)
ax3.legend(loc='lower left')
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_yticks([0,0.1])

ax4 = fig.add_subplot(gs2[1,0],sharex=ax3)
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax4.plot(ACFbins,ACF-ACFoff,lw=0.7,color='orange',label='residuals')
ax4.plot(ACFbins,lorentz(ACFbins,*popt),color='green',label=r"$t_{\rm char}$"+u' = {:.2f} \u03bcs'.format(np.abs(popt[0])))
ax4.plot(ACFbins[len(ACFbins)//2-80:len(ACFbins)//2+80],lorentz(ACFbins[len(ACFbins)//2-80:len(ACFbins)//2+80],*popt2),color='purple',label=r"$t_{\rm char}$"+u' = {:.2f} \u03bcs'.format(np.abs(popt2[0])))

ax4.set_ylabel('ACF')
ax4.set_xlim(-50,50)
ax4.set_ylim(-0.1,0.25)
ax4.legend(loc='upper right')
ax4.set_yticks([-0.1,0,0.1,0.2])

ax8 = fig.add_subplot(gs4[0,0])
point = ax8.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.7), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax8.plot(ACFbins,ACF-ACFoff,lw=0.7,color='orange')
ax8.plot(ACFbins,lorentz(ACFbins,*popt),color='green')
ax8.plot(ACFbins[len(ACFbins)//2-80:len(ACFbins)//2+80],lorentz(ACFbins[len(ACFbins)//2-80:len(ACFbins)//2+80],*popt2),color='purple')
ax8.plot(ACFbins[len(ACFbins)//2-10:len(ACFbins)//2+10],lorentz(ACFbins[len(ACFbins)//2-10:len(ACFbins)//2+10],*popt3),lw=0.5,color='cyan',label=r"$t_{\rm char}$"+u' = {:.2f} \u03bcs'.format(np.abs(popt3[0])))
ax8.set_xlabel(u'Time lag [\u03bcs]')
ax8.set_ylabel('ACF')
ax8.set_xlim(-3,3)
ax8.set_ylim(0,0.25)
ax8.legend(loc='upper left')
ax8.set_yticks([0,0.1,0.2])


ax5 = fig.add_subplot(gs3[0,0])
#load in numpy arrays of the power spectra and fits (using STINGRAY tools)
f,p=np.load('./data_products/PS_B3_IF3568_31.25ns.npy')
f8,p8=np.load('./data_products/PS_B3_IF3568_31.25ns.f8.npy')
pl =np.load('./data_products/PL_fit_B3.npy')
pl_lor=np.load('./data_products/PL_lor_fit_B3.npy')
point = ax5.scatter(1e6, 2, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax5.loglog(f,p,lw=0.7,color='#756D76', label='Average PS',linestyle='steps-mid')
ax5.loglog(f8,p8,lw=0.7,color='k',alpha=1, label='Downsample factor 8',linestyle='steps-mid')
ax5.loglog(f8,pl,lw=0.9,color='purple', label='PL')
ax5.loglog(f8,pl_lor,lw=0.9,color='deepskyblue', label='PL+Lor')
ax5.set_ylabel('Power')
ax5.legend(loc='lower left')
plt.setp(ax5.get_xticklabels(), visible=False)


ax6 = fig.add_subplot(gs3[1,0],sharex=ax5)
point = ax6.scatter(1e6, 2, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax6.loglog(f8,2*p8/pl,lw=0.7,color='purple', label='PL resid',linestyle='steps-mid')
ax6.axhline(2,color='#756D76',linestyle='--')
ax6.set_ylabel('Power')
ax6.legend()

ax7 = fig.add_subplot(gs3[2,0],sharex=ax5)
point = ax7.scatter(1e6, 2, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax7.loglog(f8,2*p8/pl_lor,lw=0.7,color='deepskyblue', label='PL+Lor resid',linestyle='steps-mid')
ax7.axhline(2,color='#756D76',linestyle='--')
ax7.set_xlabel(u'Frequency [Hz]')
ax7.set_ylabel('Power')
ax7.legend()


plt.tight_layout()

plt.savefig('edfigure4_KN.eps',format='eps',dpi=300)
plt.show()




