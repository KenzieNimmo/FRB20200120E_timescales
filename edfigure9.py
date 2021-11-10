"""
Plot of DM determination using burst B3 from M81R Nimmo et al. 2021b
Kenzie Nimmo 2021

Data is SFXC 500ns 
DM determined using 1us (downsampled factor 2) data

Data you need:
- 500ns dynamic spectrum of burst B3 -- created from a SFXC filterbank at 500ns resolution
    - pr143a_corr_no0015_500ns_1000kHz_StokesI_FullDedisp_dm87.7527_DS.npy
- DM vs peak S/N array for burst B3, and a Gaussian fit to this
    - DM_vs_peakSN_sfxc_IF1-11.npy, DM_vs_peakSN_sfxc_IF1-11_fit.npy


"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec 
from scipy.signal import correlate
import matplotlib as mpl

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'


def gaus(x,A,mu,sigma):
    return A*np.exp(-1*(x-mu)**2 / (2 * sigma**2))

#import 500ns dynamic spectrum
DS = np.load('./data_products/pr143a_corr_no0015_500ns_1000kHz_StokesI_FullDedisp_dm87.7527_DS.npy')

#downsample to 1us
DS = np.reshape(DS,(DS.shape[0],int(DS.shape[1]/2.),2)).sum(axis=2)
#IF 1 -- 11 was used to determine the DM, and so that's the profile we'll show
prof = np.mean(DS[0:177,:],axis=0)

#now downsample DS in frequency for plotting
DS = np.array(np.row_stack([np.mean(subint, axis=0) for subint in np.vsplit(DS,16)]))
#convert to SN
peak = np.argmax(prof)
offprof=prof[0:790500]
prof=prof[790500:791000]
DS = DS[:,790500:791000]
prof-=np.mean(offprof)
offprof-=np.mean(prof)
prof/=np.std(offprof)

#centre the burst
prof=np.roll(prof,int(len(prof)/2.-peak))
DS=np.roll(DS,int(len(prof)/2.-peak),axis=1)

times = np.arange(0,len(prof),1)*1.0 #microseconds
times-=(len(times)/2.)
times*=1e-3 #milliseconds

#import DM vs peak S/N
DM,peakSN=np.load('./data_products/DM_vs_peakSN_sfxc_IF1-11.npy')
#import gaus fit parameters
A,mu,sigma=np.load('./data_products/DM_vs_peakSN_sfxc_IF1-11_fit.npy')

print("DM is "+str(mu)+"+-"+str(sigma/A))


#plot
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(ncols=1, nrows=1,left=0.07,right=0.49, wspace=0.0, hspace=0.0)
gs2 = gridspec.GridSpec(ncols=1, nrows=2,left=0.58,right=0.98, height_ratios=[1,3], wspace=0.0, hspace=0.0)

cmap = plt.cm.gist_yarg

ax1 = fig.add_subplot(gs[0,0]) # DM vs peak S/N
point = ax1.scatter(87.75, 40, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax1.scatter(DM,peakSN,color='k',alpha=0.8)
ax1.plot(DM,gaus(DM,A,mu,sigma),color='r',lw=0.9)
ax1.set_xlabel(r'Dispersion Measure [pc cm$^{-3}$]')
ax1.set_ylabel('Peak S/N')
ax1.axvline(87.7527,color='k',alpha=0.5,label='DM {}'.format(87.7527))
ax1.set_xlim(87.739,87.766)
ax1.legend()

ax2 = fig.add_subplot(gs2[0,0]) # profile B3
point = ax2.scatter(0, 40, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax2.plot(times,prof,linestyle='steps-mid',label=u'1\u03bcs') 
ax2.set_xlim(-0.05,0.075)
ax2.legend()
ax2.set_ylabel('S/N')

ax3 = fig.add_subplot(gs2[1,0], sharex=ax2) # ds B3 
point = ax3.scatter(0, 1400, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), (' c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax3.imshow(DS,interpolation='nearest',aspect='auto',origin='lower',extent=(times[0],times[-1],1262,1510))
ax3.axhline(1262,color='r')
ax3.axhline(1438,color='r')
ax3.set_xlim(-0.05,0.075)
ax3.set_ylim(1261,1510)
ax3.set_xlabel('Time [ms]')
ax3.set_ylabel('Frequency [MHz]')

plt.savefig('supp_fig1_KN.eps',format='eps')
plt.show()
