"""
Burst polarisation profiles and dynamic spectra - Figure 2 from Nimmo et al. 2021b
Comparison of spectra - ED Figure 10 from Nimmo et al. 2021b

PRECISE

Kenzie Nimmo 2021

Data you need:
- calibrated archive files for the 5 bursts 
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib
    - pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib
    - pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib
    - pr158a_corr_no0017_8us_125kHz_FullPol_FullDedisp_dm87.75.ar.calib


"""
from load_file import load_archive

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pol_prof import get_profile
from parallactic_angle import parangle as pa
from astropy.coordinates import Angle

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

def subband_ticks(ds,extent,size,double=False):
    zap_size = int(ds.shape[1]/size)
    vmin=np.min(ds)
    chan_width = (extent[3]-extent[2])/(ds.shape[0]-1)
    subbands = 16. #MHz                                                                                                                                                
    nchan_per_subband = subbands/chan_width
    for subb in range(int(ds.shape[0]/nchan_per_subband)-1):
        ds[int((subb+1)*nchan_per_subband),:zap_size] = vmin-1000
        if double == True:
            ds[int((subb+1)*nchan_per_subband)+1,:zap_size] = vmin-1000
            ds[int((subb+1)*nchan_per_subband)-1,:zap_size] = vmin-1000
    return ds

"""
INPUT PARAMETERS
"""
DM = 87.75
#determined the instrumental delay by brute force searching for the delay that maximises the linear polarisation of the test pulsar with known RM - as described in the Methods of Nimmo et al. 2021b
delay_feb = -0.22
delay_march = -4.16

RM_b1 = -21.9
RM_b2 = -57.2
RM_b3 = -37.1
RM_b4 = -36.4

"""
READ IN DATA 
"""
ds_b1, extent_b1, tsamp_b1 = load_archive('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib',dm=DM,extent=True,fscrunch=8) #burst B1
ds_b2, extent_b2, tsamp_b2 = load_archive('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib',dm=DM,extent=True,fscrunch=8) #burst B2
ds_b3, extent_b3, tsamp_b3 = load_archive('./data_products/pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib',dm=DM,extent=True,fscrunch=8) #burst B3
ds_b4, extent_b4, tsamp_b4 = load_archive('./data_products/pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib',dm=DM,extent=True,fscrunch=8) #burst B4
ds_b5, extent_b5, tsamp_b5 = load_archive('./data_products/pr158a_corr_no0017_8us_125kHz_FullPol_FullDedisp_dm87.75.ar.calib',dm=DM,extent=True,fscrunch=32) #burst B5

#note B5 was not bright enough to determine the polarimetry and so only Stokes I is presented

#note we are using the .calib files (calibrated archive files), which flips alterating subbands in Stokes U (introduced in SFXC) and flips the order of RR and LL to match convention PSR/IEEE 

"""
COMPUTE PARALLACTIC ANGLE CORRECTION FOR PPA
"""
#(MJD,longitude,RA,dec,lat)
#note using the geocentric TOAs (not barycentre corrected)
parang_b1 = pa(59265.87920138888876+0.273736/(24*3600.),(06. +53./60. + 00.99425/3600.), Angle('09h57m54.721s'),Angle('68d49m01.033s'),Angle('50d31m29.39459s'))
parang_b2 = pa(59265.88216435185313+0.436768/(24*3600.),(06. +53./60. + 00.99425/3600.), Angle('09h57m54.721s'),Angle('68d49m01.033s'),Angle('50d31m29.39459s'))
parang_b3 = pa(59280.69291666666686+0.790704/(24*3600.),(06. +53./60. + 00.99425/3600.), Angle('09h57m54.721s'),Angle('68d49m01.033s'),Angle('50d31m29.39459s'))
parang_b4 = pa(59280.79847222222452+0.4208/(24*3600.),(06. +53./60. + 00.99425/3600.), Angle('09h57m54.721s'),Angle('68d49m01.033s'),Angle('50d31m29.39459s'))

"""
GET PROFILES
"""
conv = 2.355 #conversion from FWHM to sigma

I_ds_b1, I_b1, Ltrue_b1, V_b1, PA_b1, x_b1, PAmean_b1, PAerror_b1, pdf_b1, weights_b1,begin_b1,end_b1 = get_profile(ds_b1,np.linspace(extent_b1[2],extent_b1[3],ds_b1.shape[1]),delay_feb,RM_b1,parang_b1,burstbeg=811-((2*94e-6/tsamp_b1)),burstend=811+((2*94e-6/tsamp_b1)),tcent=811)
I_ds_b2, I_b2, Ltrue_b2, V_b2, PA_b2, x_b2, PAmean_b2, PAerror_b2, pdf_b2, weights_b2,begin_b2,end_b2 = get_profile(ds_b2,np.linspace(extent_b2[2],extent_b2[3],ds_b2.shape[1]),delay_feb,RM_b2,parang_b2,burstbeg=((18.85e-3/tsamp_b2)-(2*37.5e-6/tsamp_b2)),burstend=(19.09e-3/tsamp_b2)+((55.9e-6)*2/tsamp_b2),tcent=2373)
I_ds_b3, I_b3, Ltrue_b3, V_b3, PA_b3, x_b3, PAmean_b3, PAerror_b3, pdf_b3, weights_b3,begin_b3,end_b3 = get_profile(ds_b3,np.linspace(extent_b3[2],extent_b3[3],ds_b3.shape[1]),delay_march,RM_b3,parang_b3,burstbeg=(2213-(2*28e-6/tsamp_b3)),burstend=(2213+(2*28e-6/tsamp_b3)),tcent=2213)
I_ds_b4, I_b4, Ltrue_b4, V_b4, PA_b4, x_b4, PAmean_b4, PAerror_b4, pdf_b4, weights_b4,begin_b4,end_b4 = get_profile(ds_b4,np.linspace(extent_b4[2],extent_b4[3],ds_b4.shape[1]),delay_march,RM_b4,parang_b4,burstbeg=((2199)-((2*70e-6/tsamp_b4))),burstend=(2199)+((2*70e-6/tsamp_b4)),tcent=2199)
I_ds_b5, I_b5, Ltrue_b5, V_b5, PA_b5, x_b5, PAmean_b5, PAerror_b5, pdf_b5, weights_b5,begin_b5,end_b5 = get_profile(ds_b5,np.linspace(extent_b5[2],extent_b5[3],ds_b5.shape[1]),0,0,parang_b4,burstbeg=int(1857-(2*34e-6/tsamp_b5)),burstend=int(1857+(2*34e-6/tsamp_b5)),tcent=1857)

#add the red ticks indicating the subband edges for the dynamic spectra
vmin_b1 = np.min(I_ds_b1)
I_ds_b1 = subband_ticks(I_ds_b1,extent_b1,2.11)
vmin_b2 = np.min(I_ds_b2)
I_ds_b2 = subband_ticks(I_ds_b2,extent_b2,2.071)
vmin_b3 = np.min(I_ds_b3)
I_ds_b3 = subband_ticks(I_ds_b3,extent_b3,2.027)
vmin_b4 = np.min(I_ds_b4)
I_ds_b4 = subband_ticks(I_ds_b4,extent_b4,2.035)
vmin_b5 = np.min(I_ds_b5)
I_ds_b5 = subband_ticks(I_ds_b5,extent_b5,2.031)


# Extended Data Figure 10
# Compare the dpectra of the bursts
specb1 = np.mean(I_ds_b1[:,begin_b1:end_b1],axis=1)
specb2 = np.mean(I_ds_b2[:,begin_b2:end_b2],axis=1)
specb3 = np.mean(I_ds_b3[:,begin_b3:end_b3],axis=1)
specb4 = np.mean(I_ds_b4[:,begin_b4:end_b4],axis=1)
specb5 = np.mean(I_ds_b5[:,begin_b5:end_b5],axis=1)

cc1=np.corrcoef(specb1,specb2)[0,1]
cc2=np.corrcoef(specb3,specb4)[0,1]

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(211)
ax.text(-0.05,1.05,"a",transform=ax.transAxes)
point = ax.scatter(extent_b2[2], 0, facecolors='none', edgecolors='none')
plotlabel=ax.legend((point,point), (r'$\delta t \sim 4.3$ min',r'$R = %.2f$'%cc1), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax.plot(np.linspace(extent_b2[2],extent_b2[3],ds_b2.shape[1]),specb2/np.max(specb2),'#989594',label='B2')
ax.plot(np.linspace(extent_b1[2],extent_b1[3],ds_b1.shape[1]),specb1/np.max(specb2),'k',label='B1')
ax2 = fig.add_subplot(212)
ax2.text(-0.05,1.05,"b",transform=ax2.transAxes)
point = ax2.scatter(extent_b2[2], 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), (r'$\delta t \sim 2.5$ hr',r'$R = %.2f$'%cc2), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax2.plot(np.linspace(extent_b3[2],extent_b3[3],ds_b3.shape[1]),specb3,'#989594',label='B3')
ax2.plot(np.linspace(extent_b4[2],extent_b4[3],ds_b4.shape[1]),specb4,'k',label='B4')

ax2.set_xlabel('Frequency [MHz]')
ax.set_ylabel('Flux [arb. units]')
ax2.set_ylabel('Flux [arb. units]')
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)
ax.axes.yaxis.set_ticks([])
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.axes.yaxis.set_ticks([])

ax.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.savefig('supp_fig2_KN.eps',format='eps',dpi=300)
plt.show()

#bin edges of the five bursts
# centre is determined using a gaussian fit to the dynamic spectrum
# width is determined using a gaussian fit to a 2D autocorrelation of the dynamic spectrum
# this range is used for the fluence calculations
print("B1",int(811-((2*94e-6/tsamp_b1))),int(811+((2*94e-6/tsamp_b1))))
print("B2",int(((18.85e-3/tsamp_b2)-(2*37.5e-6/tsamp_b2))),int((19.09e-3/tsamp_b2)+((55.9e-6)*2/tsamp_b2)))
print("B3",int(2213-(2*28e-6/tsamp_b3)),int(2213+(2*28e-6/tsamp_b3)))
print("B4",int((2199)-((2*70e-6/tsamp_b4))),int((2199)+((2*70e-6/tsamp_b4))))
print("B5",int(1857-(2*34e-6/tsamp_b5)),int(1857+(2*34e-6/tsamp_b5)))


# Figure 2
#time arrays in microseconds
x_b1*=(tsamp_b1*1e6) #microseconds
x_b2*=(tsamp_b2*1e6) 
x_b3*=(tsamp_b3*1e6)
x_b4*=(tsamp_b4*1e6)
x_b5*=(tsamp_b5*1e6)

#apply mask
I_ds_b1=np.ma.masked_where(I_ds_b1==0,I_ds_b1)
I_ds_b2=np.ma.masked_where(I_ds_b2==0,I_ds_b2)
I_ds_b3=np.ma.masked_where(I_ds_b3==0,I_ds_b3)
I_ds_b4=np.ma.masked_where(I_ds_b4==0,I_ds_b4)
I_ds_b5=np.ma.masked_where(I_ds_b5==0,I_ds_b5)
"""
PLOT
"""
fig = plt.figure(figsize=(8, 8))
rows=3
cols=2
widths = [3, 1]
heights = [1,2,3]
gs = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.1,right=0.53,bottom=0.69,top=0.96,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs2 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.55 ,right=0.98,bottom=0.69,top=0.96,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs3 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.1,right=0.53,bottom=0.38,top=0.65,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs4 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.55 ,right=0.98,bottom=0.38,top=0.65,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs5 = gridspec.GridSpec(ncols=cols, nrows=2, left=0.1,right=0.53,bottom=0.07,top=0.34,width_ratios=widths, height_ratios=[2,3], wspace=0.0, hspace=0.0)

cmap = plt.cm.gist_yarg

ax1 = fig.add_subplot(gs[0,0]) #PA B1
ax1.axhline(PAmean_b1,color='orange',lw=0.7,zorder=0.7,alpha=0.5)
ax1.imshow(pdf_b1,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b1[0],x_b1[-1],-90,90])
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_xlim(-500,500)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_ylabel('PPA [deg]')
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0],sharex=ax1) #profile B1
ax2.step(x_b1,I_b1,'k',alpha=1.0,where='mid',lw=0.7)
ax2.step(x_b1,V_b1,'b',alpha=1.0,where='mid',lw=0.7)
ax2.step(x_b1,Ltrue_b1,'r',alpha=1.0,where='mid',lw=0.7)
ax2.set_ylabel('S/N')
point = ax2.scatter(x_b1[0], I_b1[0], facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('B1',u'8\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.995), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax2.set_xlim(-500,500)
plt.setp(ax2.get_xticklabels(), visible=False)
y_range = np.max(I_b1)-np.min(I_b1)
ax2.hlines(y=-y_range/3.,xmin=x_b1[begin_b1],xmax=x_b1[end_b1], lw=10,color='orange',zorder=0.8, alpha=0.3)
ax2.set_yticks([0,2.5,5])
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs[2,0],sharex=ax2) #ds B1
ax3.imshow(I_ds_b1,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b1[0],x_b1[-1],extent_b1[2],extent_b1[3]],vmin=vmin_b1*0.1, vmax=0.9*np.max(I_ds_b1))
cm1 = mpl.colors.ListedColormap(['black','red'])
mask=I_ds_b1<vmin_b1-600
mask = np.ma.masked_where(mask==False,mask)
ax3.imshow(mask, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[x_b1[0],x_b1[-1],extent_b1[2],extent_b1[3]])
ax3.set_xlim(-500,500)
ax3.set_ylim(extent_b1[2],extent_b1[3])
ax3.set_ylabel('Frequency [MHz]')
point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), (' j',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 1.03), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs[2,1], sharey=ax3) #spectum B1
ax4.plot(np.mean(I_ds_b1[:,begin_b1:end_b1],axis=1)/np.max(np.mean(I_ds_b1[:,begin_b1:end_b1],axis=1)),np.linspace(extent_b1[2],extent_b1[3],I_ds_b1.shape[0]), color='k',lw=0.7)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
ax4.set_ylim(extent_b1[2],extent_b1[3])
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('o',), loc='upper left',handlelength=0,bbox_to_anchor=(0.76, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax5 = fig.add_subplot(gs2[0,0]) #PA B2
ax5.imshow(pdf_b2,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b2[0],x_b2[-1],-90,90])
plt.setp(ax4.get_xticklabels(), visible=False)
ax5.set_xlim(-500,500)
plt.setp(ax5.get_xticklabels(), visible=False)
ax5.axhline(PAmean_b2,color='purple',lw=0.7,zorder=0.7,alpha=0.5)
ax5.set_ylim(PAmean_b2-20,PAmean_b2+20)
ax5.set_yticks([10,30])
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax6 = fig.add_subplot(gs2[1,0],sharex=ax5) #profile B2
ax6.step(x_b2,I_b2,'k',alpha=1.0,where='mid',lw=0.7)
ax6.step(x_b2,V_b2,'b',alpha=1.0,where='mid',lw=0.7)
ax6.step(x_b2,Ltrue_b2,'r',alpha=1.0,where='mid',lw=0.7)
point = ax6.scatter(x_b2[0], I_b2[0], facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('B2',u'8\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax6.set_xlim(-500,500)
plt.setp(ax6.get_xticklabels(), visible=False)
y_range = np.max(I_b2)-np.min(I_b2)
ax6.hlines(y=-y_range/3.,xmin=x_b2[begin_b2],xmax=x_b2[end_b2], lw=10,color='purple',zorder=0.8, alpha=0.3)
ax6.set_yticks([0,10,20,30])
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax7 = fig.add_subplot(gs2[2,0],sharex=ax6) #ds B2
ax7.imshow(I_ds_b2,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b2[0],x_b2[-1],extent_b2[2],extent_b2[3]])
mask=I_ds_b2<vmin_b2-600
mask = np.ma.masked_where(mask==False,mask)
ax7.imshow(mask, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[x_b2[0],x_b2[-1],extent_b2[2],extent_b2[3]])
ax7.set_xlim(-500,500)
plt.setp(ax7.get_yticklabels(), visible=False)
ax7.set_ylim(extent_b2[2],extent_b2[3])
point = ax7.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), (' k',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 1.03), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax8 = fig.add_subplot(gs2[2,1], sharey=ax7) #spectrum B2
ax8.plot(np.mean(I_ds_b2[:,begin_b2:end_b2],axis=1)/np.max(np.mean(I_ds_b2[:,begin_b2:end_b2],axis=1)),np.linspace(extent_b2[2],extent_b2[3],I_ds_b2.shape[0]), color='k',lw=0.7)
plt.setp(ax8.get_yticklabels(), visible=False)
plt.setp(ax8.get_xticklabels(), visible=False)
ax8.set_ylim(extent_b2[2],extent_b2[3])
point = ax8.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('p',), loc='upper left',handlelength=0,bbox_to_anchor=(0.76, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax9 = fig.add_subplot(gs3[0,0]) #PA B3                                                                                                                               
ax9.imshow(pdf_b3,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[0],x_b3[-1],-90,90])
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.set_xlim(-200,200)
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.axhline(PAmean_b3,color='green',lw=0.7,zorder=0.7,alpha=0.5)
ax9.set_ylim(PAmean_b3-20,PAmean_b3+20)
ax9.set_ylabel('PPA [deg]')
point = ax9.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)


ax10 = fig.add_subplot(gs3[1,0],sharex=ax9) #profile B3                                                                                                           
ax10.step(x_b3,I_b3,'k',alpha=1.0,where='mid',lw=0.7)
ax10.step(x_b3,V_b3,'b',alpha=1.0,where='mid',lw=0.7)
ax10.step(x_b3,Ltrue_b3,'r',alpha=1.0,where='mid',lw=0.7)
ax10.set_ylabel('S/N')
point = ax10.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('B3',u'8\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax10.set_xlim(-200,200)
plt.setp(ax10.get_xticklabels(), visible=False)
y_range = np.max(I_b3)-np.min(I_b3)
ax10.hlines(y=-y_range/3.,xmin=x_b3[begin_b3],xmax=x_b3[end_b3], lw=10,color='green',zorder=0.8, alpha=0.3)
ax10.set_yticks([0,25,50])
point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax11 = fig.add_subplot(gs3[2,0],sharex=ax10) #ds B3                                                                  
ax11.imshow(I_ds_b3,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[0],x_b3[-1],extent_b3[2],extent_b3[3]])
mask=I_ds_b3<vmin_b3-600
mask = np.ma.masked_where(mask==False,mask)
ax11.imshow(mask, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[x_b3[0],x_b3[-1],extent_b3[2],extent_b3[3]])
ax11.set_xlim(-200,200)
ax11.set_ylim(extent_b3[2],extent_b3[3])
ax11.set_ylabel('Frequency [MHz]')
point = ax11.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax11.legend((point,point), (' l',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 1.03), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax12 = fig.add_subplot(gs3[2,1], sharey=ax11) #spectum B3                                                                                                            
ax12.plot(np.mean(I_ds_b3[:,begin_b3:end_b3],axis=1)/np.max(np.mean(I_ds_b3[:,begin_b3:end_b3],axis=1)),np.linspace(extent_b3[2],extent_b3[3],I_ds_b3.shape[0]), color='k',lw=0.7)
plt.setp(ax12.get_yticklabels(), visible=False)
plt.setp(ax12.get_xticklabels(), visible=False)
ax12.set_ylim(extent_b3[2],extent_b3[3])
point = ax12.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('q',), loc='upper left',handlelength=0,bbox_to_anchor=(0.76, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax13 = fig.add_subplot(gs4[0,0]) #PA B4                                                                                                          
ax13.imshow(pdf_b4,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b4[0],x_b4[-1],-90,90])
plt.setp(ax13.get_xticklabels(), visible=False)
ax13.set_xlim(-500,500)
plt.setp(ax13.get_xticklabels(), visible=False)
ax13.axhline(PAmean_b4,color='magenta',lw=0.7,zorder=0.7,alpha=0.5)
ax13.set_ylim(PAmean_b4-20,PAmean_b4+20)
point = ax13.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax13.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax14 = fig.add_subplot(gs4[1,0],sharex=ax13) #profile B4                                                                                     
ax14.step(x_b4,I_b4,'k',alpha=1.0,where='mid',lw=0.7)
ax14.step(x_b4,V_b4,'b',alpha=1.0,where='mid',lw=0.7)
ax14.step(x_b4,Ltrue_b4,'r',alpha=1.0,where='mid',lw=0.7)
point = ax14.scatter(x_b4[0], I_b4[0], facecolors='none', edgecolors='none')
plotlabel=ax14.legend((point,point), ('B4',u'8\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax14.set_xlim(-500,500)
plt.setp(ax14.get_xticklabels(), visible=False)
y_range = np.max(I_b4)-np.min(I_b4)
ax14.hlines(y=-y_range/3.,xmin=x_b4[begin_b4],xmax=x_b4[end_b4], lw=10,color='magenta',zorder=0.8, alpha=0.3)
ax14.set_yticks([0,10,20,30])
point = ax14.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax14.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax15 = fig.add_subplot(gs4[2,0],sharex=ax14) #ds B4                                                                                                       
ax15.imshow(I_ds_b4,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b4[0],x_b4[-1],extent_b4[2],extent_b4[3]])
ax15.set_xlabel(u'Time [\u03bcs]')
mask=I_ds_b4<vmin_b4-600
mask = np.ma.masked_where(mask==False,mask)
ax15.imshow(mask, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[x_b4[0],x_b4[-1],extent_b4[2],extent_b4[3]])
ax15.set_xlim(-500,500)
plt.setp(ax15.get_yticklabels(), visible=False)
ax15.set_ylim(extent_b4[2],extent_b4[3])
point = ax15.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax15.legend((point,point), (' m',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 1.03), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax16 = fig.add_subplot(gs4[2,1], sharey=ax15) #spectrum B4                                                                                              
ax16.plot(np.mean(I_ds_b4[:,begin_b4:end_b4],axis=1)/np.max(np.mean(I_ds_b4[:,begin_b4:end_b4],axis=1)),np.linspace(extent_b4[2],extent_b4[3],I_ds_b4.shape[0]), color='k',lw=0.7)
plt.setp(ax16.get_yticklabels(), visible=False)
plt.setp(ax16.get_xticklabels(), visible=False)
ax16.set_ylim(extent_b4[2],extent_b4[3])
point = ax16.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax16.legend((point,point), ('r',), loc='upper left',handlelength=0,bbox_to_anchor=(0.76, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)


ax17 = fig.add_subplot(gs5[0,0]) #profile B5
ax17.step(x_b5,I_b5,'k',alpha=1.0,where='mid',lw=0.7)
ax17.set_ylabel('S/N')
point = ax17.scatter(x_b5[0], I_b5[0], facecolors='none', edgecolors='none')
plotlabel=ax17.legend((point,point), ('B5',u'8\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99),handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax17.set_xlim(-300,300)
plt.setp(ax17.get_xticklabels(), visible=False)
y_range = np.max(I_b5)-np.min(I_b5)
ax17.hlines(y=-y_range/3.,xmin=x_b5[begin_b5],xmax=x_b5[end_b5], lw=10,color='skyblue',zorder=0.8, alpha=0.3)
ax17.set_yticks([0,2.5,5])
point = ax17.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax17.legend((point,point), ('i',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax18 = fig.add_subplot(gs5[1,0],sharex=ax17) #ds B5
ax18.imshow(I_ds_b5,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b5[0],x_b5[-1],extent_b5[2],extent_b5[3]],vmin=vmin_b5*0.1, vmax=0.9*np.max(I_ds_b5))
cm1 = mpl.colors.ListedColormap(['black','red'])
mask=I_ds_b5<vmin_b5-600
mask = np.ma.masked_where(mask==False,mask)
ax18.imshow(mask, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[x_b5[0],x_b5[-1],extent_b5[2],extent_b5[3]])
ax18.set_xlim(-300,300)
ax18.set_ylim(extent_b5[2],extent_b5[3])
ax18.set_ylabel('Frequency [MHz]')
ax18.set_xlabel(u'Time [\u03bcs]')
point = ax18.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax18.legend((point,point), (' n',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 1.03), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax19 = fig.add_subplot(gs5[1,1], sharey=ax18) #spectum B5
ax19.plot(np.mean(I_ds_b5[:,begin_b5:end_b5],axis=1)/np.max(np.mean(I_ds_b5[:,begin_b5:end_b5],axis=1)),np.linspace(extent_b5[2],extent_b5[3],I_ds_b5.shape[0]), color='k',lw=0.7)
plt.setp(ax19.get_yticklabels(), visible=False)
plt.setp(ax19.get_xticklabels(), visible=False)
ax19.set_ylim(extent_b5[2],extent_b5[3])
point = ax19.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax19.legend((point,point), ('s',), loc='upper left',handlelength=0,bbox_to_anchor=(0.76, 1.03), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

plt.savefig('figure2_KN.pdf',format='pdf',dpi=300)
plt.show()



