"""
Script to make ED Figure 8 from Nimmo et al. 2021b
high time res polarimetry

Kenzie Nimmo 2021

Data you need:
- Calibrated archive file, 125ns full pol, containing burst B3
    - pr143a_corr_no0015_125ns_4000kHz_FullPol_FullDedisp_dm87.7527_SFXC.cor_Ef.calib


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from load_file import load_archive
from parallactic_angle import parangle as pa
from astropy.coordinates import Angle
from pol_prof import get_profile
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


fil,extent,tsamp=load_archive('./data_products/pr143a_corr_no0015_125ns_4000kHz_FullPol_FullDedisp_dm87.7527_SFXC.cor_Ef.calib',extent=True)

freqs = np.linspace(extent[2],extent[3],fil.shape[1])
#RM and delay to derotate
RM_b3 = -37.1
delay = -4.16

#chop out relevant IFs
fil=fil[:,0:44,:]
I=fil[0,:,:]+fil[1,:,:]
peak=np.argmax(np.mean(I,axis=0))

#parallactic angle
parang_b3 = pa(59280.69291666666686+0.790704/(24*3600.),(06. +53./60. + 00.99425/3600.), Angle('09h57m54.721s'),Angle('68d49m01.033s'),Angle('50d31m29.39459s'))

I_ds_b3, I_b3, Ltrue_b3, V_b3, PA_b3, x_b3, PAmean_b3, PAerror_b3, pdf_b3, weights_b3,begin_b3,end_b3 = get_profile(fil,freqs[0:44],delay,RM_b3,parang_b3,burstbeg=1800+peak-2000,burstend=2400+peak-2000,tcent=peak)
x_b3*=(tsamp*1e6)

#plot
fig = plt.figure(figsize=(8, 8))
rows=2
cols=1
heights = [1,3]
gs = gridspec.GridSpec(ncols=cols, nrows=rows,hspace=0,left=0.1,right=0.38)
gs2 = gridspec.GridSpec(ncols=cols, nrows=rows,hspace=0,left=0.40,right=0.68)
gs3 = gridspec.GridSpec(ncols=cols, nrows=rows,hspace=0,left=0.70,right=0.98)
cmap = plt.cm.gist_yarg

ax1 = fig.add_subplot(gs[0,0]) #PA B3  
ax1.axvspan(-4,6,ymin=0,ymax=0.1,color='#FCFFAB')
ax1.axvspan(12,28,ymin=0,ymax=0.1,color='#9FC7FB')
ax1.imshow(pdf_b3,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[0],x_b3[-1],-90,90])
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.axhline(PAmean_b3,color='orange',lw=0.7)
ax1.set_ylim(PAmean_b3-40,PAmean_b3+40)                                                                                                                              
ax1.set_ylabel('PPA [deg]')
ax1.set_xlim(-50,75)


point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0],sharex=ax1) #pol prof B3
ax2.axvspan(-4,6,color='#FCFFAB')
ax2.axvspan(12,28,color='#9FC7FB')       
ax2.plot(x_b3,I_b3,'k',linestyle='steps-mid',lw=0.5)
ax2.plot(x_b3,Ltrue_b3,'r',linestyle='steps-mid',lw=0.5)
ax2.plot(x_b3,V_b3,'b',linestyle='steps-mid',lw=0.5)
point = ax2.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('B3','125ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax2.set_xlim(-50,75)
ax2.set_xlabel(u'Time [\u03bcs]')
ax2.set_ylim(-6,25)

ax2.set_ylabel('S/N')

point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs2[0,0]) #PA B3                                                                                                                                
ax3.imshow(pdf_b3,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[0],x_b3[-1],-90,90])
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.axhline(PAmean_b3,color='orange',lw=0.7)
ax3.set_ylim(PAmean_b3-40,PAmean_b3+40)
ax3.set_xlim(-4,6)

point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs2[1,0],sharex=ax3) #pol prof B3                                                                                                              
ax4.axhspan(-10,-4,color='#FCFFAB')
ax4.plot(x_b3,I_b3,'k',linestyle='steps-mid',lw=0.5)
ax4.plot(x_b3,Ltrue_b3,'r',linestyle='steps-mid',lw=0.5)
ax4.plot(x_b3,V_b3,'b',linestyle='steps-mid',lw=0.5)
point = ax4.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B3','125ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax4.set_xlim(-4,6)
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.set_xlabel(u'Time [\u03bcs]')
ax4.set_ylim(-6,25)
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax5 = fig.add_subplot(gs3[0,0]) #zoom PA B3                                                                                                                          
ax5.imshow(pdf_b3,cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[0],x_b3[-1],-90,90])
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=False)
ax5.axhline(PAmean_b3,color='orange',lw=0.7)
ax5.set_ylim(PAmean_b3-40,PAmean_b3+40)
ax5.set_xlim(12,28)
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

ax6 = fig.add_subplot(gs3[1,0],sharex=ax5) #pol prof B3                                                                                                               
ax6.axhspan(-10,-4,color='#9FC7FB')
ax6.plot(x_b3,I_b3,'k',linestyle='steps-mid',lw=0.5)
ax6.plot(x_b3,Ltrue_b3,'r',linestyle='steps-mid',lw=0.5)
ax6.plot(x_b3,V_b3,'b',linestyle='steps-mid',lw=0.5)
point = ax6.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('B3','125ns'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax6.set_ylim(-6,25)
plt.setp(ax6.get_yticklabels(), visible=False)
ax6.set_xlabel(u'Time [\u03bcs]')
ax6.set_xlim(12,28)
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)

plt.savefig('edfigure8_KN.eps',format='eps',dpi=300)
plt.show()
