"""
ED Figure 1 from Nimmo et al. 2021b
Probability of single spikes in 30ns time series

Kenzie Nimmo 2021

Data you need:
- SFXC 31.25ns filterbanks of B2, B3 and B4
    - pr141a_corr_no0069_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor2_Ef.fil
    - pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil
    - pr143a_corr_no0057_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from load_file import load_filterbank
from ACF_funcs import crosscoef, autocorr,lorentz, skewness
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib as mpl
from scipy.special import gamma
from scipy.stats import chi2, expon

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'


# B2
B2high,extent,tsamp=load_filterbank('./data_products/pr141a_corr_no0069_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor2_Ef.fil')
B2high = B2high[3,:] #only brightest IF
offB2 = B2high[:10000000]
B2high/=np.std(offB2)
offB2/=np.std(offB2)
B2high= B2high[13968000:13976000] 
B2high_time=np.arange(0.,float(len(B2high)),1.)
B2high_time-=float(np.argmax(B2high))
B2high_time*=31.25e-9*1e6

#B3
B3high,extent,tsamp=load_filterbank('./data_products/pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil')
B3high = B3high[4,:] #only brightest IF
B31us = np.reshape(B3high,(len(B3high)//32,32)).sum(axis=1)
offB3 = B3high[:20000000]
B3high/=np.std(offB3)
offB3/=np.std(offB3)
B3high= B3high[np.argmax(B3high)-2000:np.argmax(B3high)+2000]   
B3high_time=np.arange(0.,float(len(B3high)),1.)
B3high_time-=float(np.argmax(B3high))
B3high_time*=31.25e-9*1e6

#B4
B4high,extent,tsamp=load_filterbank('./data_products/pr143a_corr_no0057_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil')
B4high = B4high[9,:] #only brightest IF
offB4 = B4high[:10000000]
B4high/=np.std(offB4)
offB4/=np.std(offB4)
B4high= B4high[13456000:13472000] 
B4high_time=np.arange(0.,float(len(B4high)),1.)
B4high_time-=float(np.argmax(B4high))
B4high_time*=31.25e-9*1e6

#plot
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(ncols=2, nrows=2, bottom=0.07, top=0.35,wspace=0.2, hspace=0.25)
gs2 = gridspec.GridSpec(ncols=2, nrows=2, bottom=0.38, top=0.66,wspace=0.2, hspace=0.25)
gs3 = gridspec.GridSpec(ncols=2, nrows=2, bottom=0.69, top=0.97,wspace=0.2, hspace=0.25)

ax1 = fig.add_subplot(gs3[0,0]) #B2 full 
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('B2',u'31.25 ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('1302-1318 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.65, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax1.plot(B2high_time,B2high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax1.plot(B2high_time,offB2[1000:1000+len(B2high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax1.set_xlim(-100,100)
ax1.set_ylabel('S/N')

ax2 = fig.add_subplot(gs3[1,0]) #B2 zoom
ranges=50
bn = 10
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax2.axvspan(-ranges*0.03125,ranges*0.03125,color='#D9A0E3')
ax2.plot(B2high_time,B2high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax2.plot(B2high_time,offB2[1000:1000+len(B2high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax2.set_xlim(-5,5)
ax2.set_ylabel('S/N')


ax3 = fig.add_subplot(gs3[0,1]) #B2 pdf 
point = ax3.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
peak=np.argmax(B2high)                                                                                                                             
data=np.concatenate((B2high[peak-ranges:peak],B2high[peak+1:peak+ranges])) #range of bins to use for pdf and cdf (not including bright spike that we're testing the significance of)
hist,bins=np.histogram(data,bins=bn,density=1)
bins=(bins[1:]+bins[:-1])/2
loc,scale=expon.fit(data)
rv = expon(loc=loc,scale=scale)
ax3.hist(data,bins=bn,histtype='step',facecolor='#756D76',fill=True,density=1,label=u'+/- %s \u03bcs (+/- %s bins)'%(0.03125*ranges,ranges),edgecolor='purple')
ax3.plot(np.arange(0.1,30,0.1),rv.pdf(np.arange(0.1,30,0.1)),color='purple')
ax3.legend(loc='upper center')
ax3.set_ylabel('pdf')

ax4 = fig.add_subplot(gs3[1,1],sharex=ax3) #B2 cdf
point = ax4.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('j',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
cdf=np.cumsum(hist*(bins[1]-bins[0]))
ax4.fill_between(bins,cdf,color='#756D76',step='mid')
ax4.plot(np.arange(0,30,1),rv.cdf(np.arange(0,30,1)),color='purple')
ax4.axvline(B2high[peak],color='k',linestyle='--',label='p = %.4f'%(1-rv.cdf(B2high[peak])))
ax4.legend()
ax4.set_ylabel('cdf')
#work out the 3 sigma limit for a single bin spike
val=np.argmin(np.abs(rv.cdf(np.arange(0,30,0.1))-0.997))
val=np.arange(0,30,0.1)[val]
ax2.axhline(val,xmin=-ranges*0.03125,xmax=ranges*0.03125,color='k',linestyle='--')
ax2.text(3.5,20,'1 bin')

ax5 = fig.add_subplot(gs2[0,0]) #B3 full
point = ax5.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('B3',u'31.25 ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax5.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('1318-1334 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.65, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax5.plot(B3high_time,B3high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax5.plot(B3high_time,offB3[1000:1000+len(B3high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax5.set_xlim(-60,60)
ax5.set_ylabel('S/N')

ax6 = fig.add_subplot(gs2[1,0]) #B3 zoom
ranges=50
bn = 10
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax6.axvspan(-ranges*0.03125,ranges*0.03125,color='#96D5A7')
ax6.plot(B3high_time,B3high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax6.plot(B3high_time,offB3[1000:1000+len(B3high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax6.set_xlim(-5,5)
ax6.set_ylabel('S/N')
ax6.text(3.5,32,'1 bin')
ax6.text(3.5,18,'2 bin')

ax7 = fig.add_subplot(gs2[0,1]) #B3 pdf
point = ax7.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
peak=np.argmax(B3high)
data=np.concatenate((B3high[peak-ranges:peak],B3high[peak+2:peak+ranges]))
hist,bins=np.histogram(data,bins=bn,density=1)
bins=(bins[1:]+bins[:-1])/2
loc,scale=expon.fit(data)
rv = expon(loc=loc,scale=scale)
ax7.hist(data,bins=bn,histtype='step',facecolor='#756D76',fill=True,density=1,label=u'+/- %s \u03bcs (+/- %s bins)'%(0.03125*ranges,ranges),edgecolor='green')
ax7.set_ylabel('pdf')
ax7.plot(np.arange(0.1,50,0.1),rv.pdf(np.arange(0.1,50,0.1)),color='green')
ax7.legend(loc='upper center')

ax8 = fig.add_subplot(gs2[1,1],sharex=ax7) #B3 cdf
point = ax8.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('k',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
cdf=np.cumsum(hist*(bins[1]-bins[0]))
ax8.fill_between(bins,cdf,color='#756D76',step='mid')      
ax8.plot(np.arange(0,50,1),rv.cdf(np.arange(0,50,1)),color='green')
ax8.axvline(B3high[peak],color='k',linestyle='--',label='p = %.4f'%(1-rv.cdf(B3high[peak])))
ax8.axvline(B3high[peak+1],color='#756D76',linestyle='--',label='p = %.4f'%(1-rv.cdf(B3high[peak+1])))
ax8.legend()
ax8.set_ylabel('cdf')

val=np.argmin(np.abs(rv.cdf(np.arange(0,30,0.1))-0.997)) #3sigma one bin features
val=np.arange(0,30,0.1)[val]
ax6.axhline(val,xmin=-ranges*0.03125,xmax=ranges*0.03125,color='k',linestyle='--')

#p value of 3 sigma 1 bin is (1-0.997)
percent_2bin = 1-np.sqrt(1-0.997)
val=np.argmin(np.abs(rv.cdf(np.arange(0,30,0.1))-percent_2bin)) #3sigma 2 bin features
val=np.arange(0,30,0.1)[val]
ax6.axhline(val,xmin=-ranges*0.03125,xmax=ranges*0.03125,color='k',linestyle='--')

ax9 = fig.add_subplot(gs[0,0]) # B4 full
point = ax9.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('B4',u'31.25 ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.85), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax9.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('1398-1414 MHz',), loc='upper left',handlelength=0,bbox_to_anchor=(0.65, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax9.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax9.plot(B4high_time,B4high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax9.plot(B4high_time,offB4[1000:1000+len(B4high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax9.set_xlim(-200,200)
ax9.set_ylabel('S/N')

ax10 = fig.add_subplot(gs[1,0]) # B4 zoom
ranges=50
bn = 10
point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.005, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax10.axvspan(-ranges*0.03125,ranges*0.03125,color='#FFD4EA')
ax10.plot(B4high_time,B4high,color='#756D76',linestyle='steps-mid',lw=0.6)
ax10.plot(B4high_time,offB4[1000:1000+len(B4high)],color='k',linestyle='steps-mid',lw=0.6,alpha=1)
ax10.set_xlim(-5,5)
ax10.set_ylabel('S/N')
ax10.set_xlabel(u'Time [\u03bcs]')

ax11 = fig.add_subplot(gs[0,1]) # B4 pdf
point = ax11.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax11.legend((point,point), ('i',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
peak=np.argmax(B4high)
data=np.concatenate((B4high[peak-ranges:peak],B3high[peak+1:peak+ranges]))
hist,bins=np.histogram(data,bins=bn,density=1)
bins=(bins[1:]+bins[:-1])/2
loc,scale=expon.fit(data)
rv = expon(loc=loc,scale=scale)
ax11.hist(data,bins=bn,histtype='step',facecolor='#756D76',fill=True,density=1,label=u'+/- %s \u03bcs (+/- %s bins)'%(0.03125*ranges,ranges),edgecolor='pink')
ax11.plot(np.arange(0.1,20,0.1),rv.pdf(np.arange(0.1,20,0.1)),color='pink')
ax11.legend(loc='upper center')
ax11.set_ylabel('pdf')

ax12 = fig.add_subplot(gs[1,1],sharex=ax11) # B4 cdf
point = ax12.scatter(0, 0.1, facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('l',), loc='upper left',handlelength=0,bbox_to_anchor=(0.94, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
cdf=np.cumsum(hist*(bins[1]-bins[0]))
ax12.fill_between(bins,cdf,color='#756D76',step='mid')
ax12.plot(np.arange(0,20,1),rv.cdf(np.arange(0,20,1)),color='pink')
ax12.axvline(B4high[peak],color='k',linestyle='--',label='p = %.4f'%(1-rv.cdf(B4high[peak])))
ax12.legend()
ax12.set_ylabel('cdf')
ax12.set_xlabel('S/N')

val=np.argmin(np.abs(rv.cdf(np.arange(0,20,0.1))-0.997)) #1 bin 3 sigma
val=np.arange(0,20,0.1)[val]
ax10.axhline(val,xmin=-ranges*0.03125,xmax=ranges*0.03125,color='k',linestyle='--')
ax10.text(3.5,13.5,'1 bin')
plt.savefig('edfigure1_KN.eps',format='eps',dpi=300)
plt.show()
