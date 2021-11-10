"""
Script to make ED Figure 7 Nimmo et al. 2021b
1us profiles and correlation coefficients

Kenzie Nimmo 2021

Data you need:
- SFXC 1us filterbanks pf B2, B3 and B4
    - pr141a_corr_no0069_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor2_Ef.fil
    - pr143a_corr_no0015_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil
    - pr143a_corr_no0057_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil



"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ACF_funcs import crosscoef
from load_file import load_filterbank
import matplotlib as mpl
from radiometer import radiometer

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'

B4,extent_B4,tsamp_B4 = load_filterbank('./data_products/pr143a_corr_no0057_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil')
B2,extent_B2,tsamp_B2 = load_filterbank('./data_products/pr141a_corr_no0069_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor2_Ef.fil')
B3, extent_B3, tsamp_B3 = load_filterbank('./data_products/pr143a_corr_no0015_1us_500kHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil')

#cut out burst in time and frequency
#B2 IF 4 to IF 13 
B2_burst=B2[3*32:14*32,3491000//8:3495000//8]
#B4 IF 4 to 14
B4_burst=B4[3*32:15*32,3364000//8:3369000//8]
#B3 IF 1 to 11
B3_burst=B3[0:32*12,790689-200:790689+3000]

#convert to S/N
B2offspec=np.mean(B2[3*32:14*32,10000:10000+B2_burst.shape[1]],axis=1)
B2off = np.mean(B2[3*32:14*32,10000:10000+B2_burst.shape[1]],axis=0)
B2_burst_prof =np.mean(B2_burst,axis=0)
B2_burst_prof-=np.mean(B2off)
B2off-=np.mean(B2off)
B2_burst_prof/=np.std(B2off)
B2off/=np.std(B2off)
B2_times = np.arange(0.,float(len(B2_burst_prof)),1.)*1e-6*1e6

B4offspec = np.mean(B4[3*32:15*32,10000:10000+B4_burst.shape[1]],axis=1)
B4off =np.mean(B4[3*32:15*32,10000:10000+B4_burst.shape[1]],axis=0)
B4_burst_prof =np.mean(B4_burst,axis=0)
B4_burst_prof-=np.mean(B4off)
B4off-=np.mean(B4off)
B4_burst_prof/=np.std(B4off)
B4_times = np.arange(0.,float(len(B4_burst_prof)),1.)*1e-6*1e6

B3offspec=np.mean(B3[0:48*8,10000:10000+B3_burst.shape[1]],axis=1)
B3off = np.mean(B3[0:48*8,10000:10000+B3_burst.shape[1]],axis=0)
B3_burst_prof =np.mean(B3_burst,axis=0)
B3_burst_prof-=np.mean(B3off)
B3off-=np.mean(B3off)
B3_burst_prof/=np.std(B3off)
B3off/=np.std(B3off)
B3_times = np.arange(0.,float(len(B3_burst_prof)),1.)*1e-6*1e6

#correlation coefficients
posB2,corrcoefs_B2,meansB2,meanval_B2 = crosscoef(B2_burst,B2_burst_prof,9,B2[3*32:14*32,10000:10000+B2_burst.shape[1]],returnvals=True)
posB4,corrcoefs_B4,meansB4,meanval_B4 = crosscoef(B4_burst,B4_burst_prof,9,B4[3*32:15*32,10000:10000+B4_burst.shape[1]],returnvals=True)
posB3,corrcoefs_B3,meansB3,meanval_B3 = crosscoef(B3_burst,B3_burst_prof,9,B3[0:48*8,10000:10000+B3_burst.shape[1]],returnvals=True)

#make histogram bins
hist,bins=np.histogram(corrcoefs_B2,bins=30)
bins=(bins[1:]+bins[:-1])/2
hist,bins4=np.histogram(corrcoefs_B4,bins=30)
bins4=(bins4[1:]+bins4[:-1])/2
hist,bins3=np.histogram(corrcoefs_B3,bins=30)
bins3=(bins3[1:]+bins3[:-1])/2

#plot
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(ncols=1, nrows=3, left=0.07, right=0.47,wspace=0.0, hspace=0.3)
gs2 = gridspec.GridSpec(ncols=2, nrows=3, left=0.55,right=0.95, wspace=0.0, hspace=0.3,width_ratios=[3,1])


ax1 = fig.add_subplot(gs[0,0]) # burst B2
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')                                                                                                        
plotlabel=ax1.legend((point,point), ('B2',u'1 \u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('1302-1462 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.7, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax1.plot(B2_times,B2_burst_prof,color='k',linestyle='steps-mid',lw=0.6)
ax1.set_ylabel('S/N')
ax1.set_xlim(0,500)
ax1.set_yticks([0,5,10])

ax4 = fig.add_subplot(gs[1,0]) # burst B3
point = ax4.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B3',u'1us'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax4.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('1254-1430 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.7, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax4.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)



ax4.plot(B3_times,B3_burst_prof,color='k',linestyle='steps-mid',lw=0.6)
ax4.set_ylabel('S/N')
ax4.set_xlim(0,400)

ax7 = fig.add_subplot(gs[2,0]) # burst B4
point = ax7.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('B4',u'1us'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax7.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('1302-1478 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.7, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax7.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax7.plot(B4_times,B4_burst_prof,linestyle='steps-mid',lw=0.6,color='k')
ax7.set_ylabel('S/N')
ax7.set_xlim(0,600)
ax7.set_xlabel(u'Time [\u03bcs]') 

ax10 = fig.add_subplot(gs2[0,0]) # corr coefs vs time lag B2
ax10.scatter(posB2,corrcoefs_B2,marker='x',c=meansB2,cmap='binary',alpha=1.0,s=0.7)
ax10.axhline(meanval_B2, color='purple')
ax10.set_ylabel('Correlation coefficient')
point = ax10.scatter(10, 0.3, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax11 = fig.add_subplot(gs2[0,1],sharey=ax10) # corr coefs hist B2   
ax11.hist(corrcoefs_B2,bins, color='#756D76',label='B2 S/N > 7',orientation='horizontal')
ax11.axhline(meanval_B2, color='purple')
plt.setp(ax11.get_yticklabels(), visible=False)

ax12 = fig.add_subplot(gs2[1,0]) # corr coefs vs time lag B3                                                    
ax12.scatter(posB3,corrcoefs_B3,marker='x',c=meansB3,cmap='binary',alpha=1.0,s=0.7)
ax12.axhline(meanval_B3, color='green')
ax12.set_ylabel('Correlation coefficient')
point = ax12.scatter(10, 0.3, facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax13 = fig.add_subplot(gs2[1,1],sharey=ax12) # corr coefs hist B3                                            
ax13.hist(corrcoefs_B3,bins3, color='#756D76',label='B3 S/N > 10',orientation='horizontal')
ax13.axhline(meanval_B3, color='green')
plt.setp(ax13.get_yticklabels(), visible=False)

ax14 = fig.add_subplot(gs2[2,0]) # corr coefs vs time lag B4
ax14.scatter(posB4,corrcoefs_B4,marker='x',c=meansB4,cmap='binary',alpha=1.0,s=0.7)
ax14.axhline(meanval_B4, color='magenta')
ax14.set_ylabel('Correlation coefficient')
ax14.set_xlabel(u'Time lag [\u03bcs]')
point = ax14.scatter(10, 0.3, facecolors='none', edgecolors='none')
plotlabel=ax14.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax15 = fig.add_subplot(gs2[2,1],sharey=ax14) # corr coefs hist B2                                            
ax15.hist(corrcoefs_B4,bins4, color='#756D76',label='B4 S/N > 6',orientation='horizontal')
ax15.axhline(meanval_B4, color='magenta')
plt.setp(ax15.get_yticklabels(), visible=False)


plt.savefig('edfigure7_KN.eps',format='eps',dpi=300)
plt.show()

