"""
Figure 1 from Nimmo et al. 2021b on the burst properties of FRB 20200120E

PRECISE


Kenzie Nimmo 2021


Data you need:
 - SFXC 31.25ns filterbank of burst B3 --> pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil


"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from load_file import load_filterbank
from radiometer import radiometer 
from scipy.stats import expon
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib as mpl

mpl.rcParams['font.size'] = 6
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.pad']='4'
mpl.rcParams['ytick.major.pad']='4'

#load in data
fil,extent,tsamp=load_filterbank('./data_products/pr143a_corr_no0015_31.25ns_16MHz_StokesI_FullDedisp_dm87.7527_SFXC.cor_Ef.fil') #SFXC

#only IF5 - where the bright scintle is in burst B3
prof = fil[4,:]

offprof = prof[:20000000]
prof/=np.std(offprof)
offprof/=np.std(offprof)

#make a time axis in microseconds
time = np.arange(0.,float(len(prof)),1)
peak=float(np.argmax(prof))
time-=peak
time*=tsamp
time*=1000000 #microsecond


#make a 1us time series
while len(prof) % 32 !=0:
    prof=prof[:-1]
    prof_flux=prof_flux[:-1]
    time=time[:-1]
prof_1us = np.reshape(prof,(len(prof)/32,32)).sum(axis=1)
time_1us=time[15::32]

#convert 1us profile to S/N
off=prof_1us[0:int(20000000/32)]
prof_1us-=np.mean(off)
off-=np.mean(off)
prof_1us/=np.std(off)

#plotting window edges
begin = np.argmax(prof)-10000
end = np.argmax(prof)+10000

#convert to Jy units
prof_flux=prof*radiometer(tsamp*1000, 16, 2, 15.454545454545455)

#fit the distribution with chi^2 2dof
def chi2_kn(x,loc,scale):
    dof = 2
    return (np.exp(-(x-loc)/scale)/scale * x**((dof/2) -1)) / (2**(dof/2) * gamma(dof/2))

def chi2_mult(x,*inp):
    outp=np.zeros_like(x)
    for i in range(len(inp)//2):
        outp+=chi2_kn(x,inp[i],inp[i + len(inp)//2])
    return outp


#plot 
fig = plt.figure(figsize=(5, 7))
rows=4 
cols=3
widths = [6,2,1]
heights = [1,1,1,1]
gs = gridspec.GridSpec(ncols=cols, nrows=rows,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.2)

ax1 = fig.add_subplot(gs[0,0]) # full profile
ax1.axvspan(-20,50,color='k',alpha=0.1)
ax1.fill_between(time_1us[begin//32:end//32], 0, prof_1us[begin//32:end//32],color='green',alpha=0.3,step='mid',label=u'1 \u03bcs')
ax1.plot(time[begin:end],prof[begin:end],linestyle='steps-mid',lw=0.4,color='k',label='31.25 ns')
ax1.set_xlim((-60,80))
ax1.set_ylim(-3,47)
ax1.set_ylabel('S/N')
ax1.legend()
ax1.set_xticks([-60,-40,-20,0,20,40,60])
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax5 = fig.add_subplot(gs[0,1],sharey=ax1) #histogram of on burst values
beginb=np.argmin(np.abs(time+20))
endb=np.argmin(np.abs(time-50))
hist,bins=np.histogram(prof[beginb:endb], bins=np.linspace(0,50,50))
histoff,binsoff=np.histogram(prof[0:endb-beginb], bins=np.linspace(0,50,50))
bins=(bins[0:-1]+bins[1:])/2.
binsoff=(binsoff[0:-1]+binsoff[1:])/2.
ax5.hist(prof[beginb:endb], bins=np.linspace(0,50,50), orientation="horizontal",color='k',alpha=0.3, label='on burst')
ax5.hist(prof[0:endb-beginb], bins=np.linspace(0,50,50), orientation="horizontal",color='red',alpha=0.3,label='off burst')
ax5.set_ylim(-3,47)
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

#fitting
loc,scale=expon.fit(prof[beginb:endb])
popt,pcov = curve_fit(chi2_mult,bins,hist,p0=[(loc,scale)])
popt2,pcov2 = curve_fit(chi2_mult,bins,hist,p0=[(loc,loc,loc,scale,scale,scale)])
loc,scale=expon.fit(prof[0:endb-beginb])
poptoff,pcovoff = curve_fit(chi2_mult,binsoff,histoff,p0=[(loc,scale)])
ax5.plot(chi2_mult(np.arange(binsoff[0],binsoff[-1],0.1),*poptoff),np.arange(binsoff[0],binsoff[-1],0.1),color='r')
ax5.set_xscale('log')
ax5.legend()
ax5.set_xlim(0.1,2e3)
plt.setp(ax5.get_yticklabels(), visible=False)

#uncertainties on off burst resids

alll=[]
loc,scale=expon.fit(offprof[0:(endb-beginb)])
for i in range(2000):
    histoffi,binsoffi=np.histogram(offprof[0 + i*(endb-beginb):(i+1)* (endb-beginb)], bins=np.linspace(0,50,50))
    binsoffi=(binsoffi[0:-1]+binsoffi[1:])/2.
    poptoffi,pcovoffi = curve_fit(chi2_mult,binsoffi,histoffi,p0=[(loc,scale)])
    resids=histoffi - chi2_mult(binsoffi,*poptoffi)
    alll = np.append(alll,resids)

errors=np.zeros_like(binsoffi)
for i in range(len(binsoffi)):
    errors[i] = np.std(alll[i::len(binsoffi)])

ax6 = fig.add_subplot(gs[0,2],sharey=ax1) #resids
offresids=histoff-chi2_mult(binsoff,*poptoff)
ax6.errorbar(histoff-chi2_mult(binsoff,*poptoff),binsoff,xerr=errors,color='r',ms=1,fmt='o',lw=0.4)
plt.setp(ax6.get_yticklabels(), visible=False)
ax6.set_ylim(-3,47)
ax6.set_xlim(-13,13)
ax6.set_xticks((-10,0,10))
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0:]) # zoom in on burst whole 
ax2.axvspan(-4,1,color='gold',alpha=0.3)
ax2.axvspan(8,20,color='royalblue',alpha=0.3)
ax2.fill_between(time_1us[begin//32:end//32], 0, prof_1us[begin//32:end//32],color='green',alpha=0.3,step='mid')
ax2.plot(time[begin:end],prof[begin:end],linestyle='steps-mid',lw=0.4,color='k')
ax2.set_xlim((-20,50))
ax2.set_ylabel('S/N')
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs[2,0:]) # zoom in again one chunk
ax3.axhspan(-30,-2,color='gold',alpha=0.3)
ax3.plot(time[begin:end],prof_flux[begin:end],linestyle='steps-mid',lw=0.4,color='k')
ax3.set_xlim((-4,1))
ax3.set_ylim((-30,700))
ax3.set_ylabel('Flux Density [Jy]')
point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs[3,0:]) # second chunk
ax4.axhspan(-30,-2,color='royalblue',alpha=0.3)
ax4.plot(time[begin:end],prof_flux[begin:end],linestyle='steps-mid',lw=0.4,color='k')
ax4.set_xlim((8,20))
ax4.set_ylim((-30,500))
ax4.set_ylabel('Flux Density [Jy]')
ax4.set_xlabel(u'Time [\u03bcs]')
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

plt.savefig('figure1_KN.pdf',format='pdf',dpi=200)
plt.show()

