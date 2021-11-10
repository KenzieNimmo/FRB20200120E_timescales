"""
Script to make polarisation plot in M81R/FRB 200120 paper. Nimmo et al. 2021b
ED Figure 11

Kenzie Nimmo 2021

Data you need:
- Faraday spectra of B1-B4
    - no0069_8us_125kHz_faradayspec.npy
    - no0069_8us_125kHz_faradayspec_2.npy
    - no0015_8us_125kHz_faradayspec.npy
    - no0057_8us_125kHz_faradayspec.npy
- QU spectra, with frequency array and S/N weights
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib_QUdata.npy
    - pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy
    - pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy
- QU joint fit results
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy
    - pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib_QUfit.npy
    - pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy
- pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
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

#load in the faraday spectra for the four bursts
RMs_b1,b1_faraday_spec = np.load('./data_products/no0069_8us_125kHz_faradayspec.npy')
RMs_b2,b2_faraday_spec = np.load('./data_products/no0069_8us_125kHz_faradayspec_2.npy')
RMs_b3,b3_faraday_spec = np.load('./data_products/no0015_8us_125kHz_faradayspec.npy')
RMs_b4,b4_faraday_spec = np.load('./data_products/no0057_8us_125kHz_faradayspec.npy')

# load in the QU data and fits
freq_b1,Qspec_b1,Uspec_b1,weights_b1 = np.load('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy')
freq_b2,Qspec_b2,Uspec_b2,weights_b2 = np.load('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib_QUdata.npy')
freq_b3,Qspec_b3,Uspec_b3,weights_b3 = np.load('./data_products/pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy')
freq_b4,Qspec_b4,Uspec_b4,weights_b4 = np.load('./data_products/pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUdata.npy')

Qfit_b1,Ufit_b1 = np.load('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy')
Qfit_b2,Ufit_b2 = np.load('./data_products/pr141a_corr_no0069_8us_125kHz_FullPol_FullDedisp_dm87.75.cor2_Ef.ar.calib_QUfit.npy')
Qfit_b3,Ufit_b3 = np.load('./data_products/pr143a_corr_no0015_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy')
Qfit_b4,Ufit_b4 = np.load('./data_products/pr143a_corr_no0057_8us_125kHz_FullPol_FullDedisp_dm87.75.cor_Ef.ar.calib_QUfit.npy')

#plotting
fig = plt.figure(figsize=(8, 8))
cmap=plt.cm.get_cmap('binary')
gs = gridspec.GridSpec(ncols=1, nrows=1, top=0.98, bottom=0.72)
gs2 = gridspec.GridSpec(ncols=2, nrows=2, top=0.65, bottom=0.39,wspace=0.15,hspace=0)
gs3 = gridspec.GridSpec(ncols=2, nrows=2, top=0.35, bottom=0.09,wspace=0.15,hspace=0)

ax1 = fig.add_subplot(gs[0,0]) # the Faraday spectra
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax1.plot(RMs_b1,b1_faraday_spec/np.max(b1_faraday_spec),color='orange',label='B1')
ax1.plot(RMs_b2,b2_faraday_spec/np.max(b2_faraday_spec),color='purple',label='B2')
ax1.plot(RMs_b3,b3_faraday_spec/np.max(b3_faraday_spec),color='green',label='B3')
ax1.plot(RMs_b4,b4_faraday_spec/np.max(b4_faraday_spec),color='magenta',label='B4')
ax1.set_xlim(-500,500)
ax1.axvline(-29.8,color='k',alpha=0.3,label='Bhardwaj et al. 2021')
ax1.axvline(-26.8,color='k',alpha=0.3)
ax1.legend()
ax1.set_ylabel('Normalised polarised intensity')
ax1.set_xlabel(r'Rotation Measure (rad/m$^2$)')

ax2 = fig.add_subplot(gs2[0,0]) # B1 Qspec
point = ax2.scatter(1400,0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), (' b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax2.scatter(freq_b1,Qspec_b1,c=weights_b1,cmap=cmap)
ax2.plot(freq_b1,Qfit_b1,color='orange')
ax2.set_ylabel('Q/L')
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = fig.add_subplot(gs2[1,0],sharex=ax2) # B1 Uspec
ax3.scatter(freq_b1,Uspec_b1,c=weights_b1,cmap=cmap)
ax3.plot(freq_b1,Ufit_b1,color='orange')
ax3.set_ylabel('U/L')

ax4 = fig.add_subplot(gs2[0,1]) # B2 Qspec                                                                           
point = ax4.scatter(1400,0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), (' c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)                                   
ax4.scatter(freq_b2,Qspec_b2,c=weights_b2,cmap=cmap)
ax4.plot(freq_b2,Qfit_b2,color='purple')
plt.setp(ax4.get_xticklabels(), visible=False)

ax5 = fig.add_subplot(gs2[1,1],sharex=ax4) # B2 Uspec    
ax5.scatter(freq_b2,Uspec_b2,c=weights_b2,cmap=cmap)
ax5.plot(freq_b2,Ufit_b2,color='purple')

ax6 = fig.add_subplot(gs3[0,0])# B3 Qspec                                                                            
point = ax6.scatter(1400,0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), (' d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax6.scatter(freq_b3,Qspec_b3,c=weights_b3,cmap=cmap)
ax6.plot(freq_b3,Qfit_b3,color='green')
ax6.set_ylabel('Q/L')
plt.setp(ax6.get_xticklabels(), visible=False)

ax7 = fig.add_subplot(gs3[1,0],sharex=ax6)# B3 Uspec                                                                                                                 
ax7.scatter(freq_b3,Uspec_b3,c=weights_b3,cmap=cmap)
ax7.plot(freq_b3,Ufit_b3,color='green')
ax7.set_ylabel('U/L')

ax8 = fig.add_subplot(gs3[0,1]) # B4 Qspec
point = ax8.scatter(1400,0, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), (' e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.99), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=8)
plt.gca().add_artist(plotlabel)
ax8.scatter(freq_b4,Qspec_b4,c=weights_b4,cmap=cmap)
ax8.plot(freq_b4,Qfit_b4,color='magenta')
plt.setp(ax8.get_xticklabels(), visible=False)


ax9 = fig.add_subplot(gs3[1,1],sharex=ax8) # B4 Uspec 
ax9.scatter(freq_b4,Uspec_b4,c=weights_b4,cmap=cmap)
ax9.plot(freq_b4,Ufit_b4,color='magenta')


fig.text(0.5, 0.04, 'Frequency [MHz]', ha='center')

plt.savefig('supp_fig3_KN.eps',format='eps',dpi=300)
plt.show()
