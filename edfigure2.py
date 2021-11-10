"""
Script to make ACF plot for M81R burst properties paper -- ED Figure 2 Nimmo et al. 2021b


Kenzie Nimmo 2021

Data you need:
- 2D ACF of the dynamic spectra of each burst
    - ACF_b*_8us_f8.npy
- The 2D Gaussian fit to the ACF
    - fitACF_b*_8us_f8.npy


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# load in ACFs -- determined using function autocorr_2D in ACF_funcs
ACF_b1 = np.load('./data_products/ACF_b1_8us_f8.npy')
ACF_b1 = np.ma.masked_where(ACF_b1==np.max(ACF_b1),ACF_b1)
times_b1,freqs_b1,ACF_fit_b1,lor_b1 = np.load('./data_products/fitACF_b1_8us_f8.npy')

ACF_b2 = np.load('./data_products/ACF_b2_8us_f8.npy')
ACF_b2 = np.ma.masked_where(ACF_b2==np.max(ACF_b2),ACF_b2)
times_b2,freqs_b2,ACF_fit_b2,lor_b2 = np.load('./data_products/fitACF_b2_8us_f8.npy')

ACF_b3 = np.load('./data_products/ACF_b3_8us_f8.npy')
ACF_b3 = np.ma.masked_where(ACF_b3==np.max(ACF_b3),ACF_b3)
times_b3,freqs_b3,ACF_fit_b3,lor_b3 = np.load('./data_products/fitACF_b3_8us_f8.npy')

ACF_b4 = np.load('./data_products/ACF_b4_8us_f8.npy')
ACF_b4 = np.ma.masked_where(ACF_b4==np.max(ACF_b4),ACF_b4)
times_b4,freqs_b4,ACF_fit_b4,lor_b4 = np.load('./data_products/fitACF_b4_8us_f8.npy')

ACF_b5 = np.load('./data_products/ACF_b5_8us_f8.npy')
ACF_b5 = np.ma.masked_where(ACF_b5==np.max(ACF_b5),ACF_b5)
times_b5,freqs_b5,ACF_fit_b5,lor_b5 = np.load('./data_products/fitACF_b5_8us_f8.npy')

#plot
fig = plt.figure(figsize=(7, 7))
rows=2
cols=2
widths = [3, 1,]
heights = [1,3]
gs = gridspec.GridSpec(ncols=cols, nrows=rows,left=0.1,right=0.53,bottom=0.70,top=0.98,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs2 = gridspec.GridSpec(ncols=cols, nrows=rows,left=0.55,right=0.98,bottom=0.70,top=0.98,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs3 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.1,right=0.53,bottom=0.38,top=0.66,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs4 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.55 ,right=0.98,bottom=0.38,top=0.66,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs5 = gridspec.GridSpec(ncols=cols, nrows=rows, left=0.1,right=0.53,bottom=0.06,top=0.34,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
cmap = plt.cm.gist_yarg

ax1 = fig.add_subplot(gs[0,0]) # Time ACF  B1
plt.setp(ax1.get_yticklabels(), visible=False)                                                                                        
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.plot(times_b1,np.mean(ACF_b1,axis=0),color='k', lw=0.7)
ax1.plot(times_b1,np.mean(ACF_fit_b1,axis=0),color='orange')
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('B1',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0],sharex=ax1) # 2D ACF B1  
ax2.imshow(ACF_b1,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times_b1[0],times_b1[-1],freqs_b1[0],freqs_b1[-1]))
T,F=np.meshgrid(times_b1, freqs_b1)
ax2.contour(T,F,ACF_fit_b1,4, colors='orange', linewidths=.5)
ax2.set_ylabel('Frequency Lag [MHz]')
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), (' f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 0.99), handletextpad=-0.5,frameon=True,markerscale=0)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs[1,1],sharey=ax2) #Freq ACF B1
ax3.plot(np.mean(ACF_b1,axis=1),freqs_b1,color='k', lw=0.7)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_ylim(freqs_b1[0],freqs_b1[-1])
ax3.plot(np.mean(ACF_fit_b1,axis=1),freqs_b1,color='orange')
point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('k',), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs2[0,0]) # Time ACF B2                                                             
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
ax4.plot(times_b2,np.mean(ACF_b2,axis=0),color='k', lw=0.7)
ax4.plot(times_b2,np.mean(ACF_fit_b2,axis=0),color='purple')

point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B2',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

    
ax5 = fig.add_subplot(gs2[1,0],sharex=ax4) # 2D ACF B2                                                                                                                
ax5.imshow(ACF_b2,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times_b2[0],times_b2[-1],freqs_b2[0],freqs_b2[-1]))
T,F=np.meshgrid(times_b2, freqs_b2)  
ax5.contour(T,F,ACF_fit_b2,4, colors='purple', linewidths=.5)
plt.setp(ax5.get_yticklabels(), visible=False) 
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), (' g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 0.99), handletextpad=-0.5,frameon=True,markerscale=0)
plt.gca().add_artist(plotlabel) 

ax6 = fig.add_subplot(gs2[1,1],sharey=ax5) #Freq ACF B2
ax6.plot(np.mean(ACF_b2,axis=1),freqs_b2,color='k', lw=0.7)
plt.setp(ax6.get_yticklabels(), visible=False)
plt.setp(ax6.get_xticklabels(), visible=False)
ax6.set_ylim(freqs_b2[0],freqs_b2[-1])
ax6.plot(np.mean(ACF_fit_b2,axis=1),freqs_b2,color='purple')
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('l',), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax7 = fig.add_subplot(gs3[0,0]) # Time ACF B3                                                           
plt.setp(ax7.get_yticklabels(), visible=False)
plt.setp(ax7.get_xticklabels(), visible=False)
ax7.plot(times_b3,np.mean(ACF_b3,axis=0),color='k', lw=0.7)
ax7.plot(times_b3,np.mean(ACF_fit_b3,axis=0),color='green')

point = ax7.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('B3',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax7.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax8 = fig.add_subplot(gs3[1,0],sharex=ax7) # 2D ACF B3                                  
ax8.imshow(ACF_b3,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times_b3[0],times_b3[-1],freqs_b3[0],freqs_b3[-1]))
T,F=np.meshgrid(times_b3, freqs_b3)
ax8.contour(T,F,ACF_fit_b3,4, colors='green', linewidths=.5)
ax8.set_ylabel('Frequency Lag [MHz]')
point = ax8.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), (' h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 0.99), handletextpad=-0.5,frameon=True,markerscale=0)
plt.gca().add_artist(plotlabel)

ax9 = fig.add_subplot(gs3[1,1],sharey=ax8) #Freq ACF B3                                                     
ax9.plot(np.mean(ACF_b3,axis=1),freqs_b3,color='k', lw=0.7)
plt.setp(ax9.get_yticklabels(), visible=False)
plt.setp(ax9.get_xticklabels(), visible=False)
ax9.set_ylim(freqs_b3[0],freqs_b3[-1])
ax9.plot(np.mean(ACF_fit_b3,axis=1),freqs_b3,color='green')
point = ax9.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('m',), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax10 = fig.add_subplot(gs4[0,0]) # Time ACF B4                                                               
plt.setp(ax10.get_yticklabels(), visible=False)
plt.setp(ax10.get_xticklabels(), visible=False)
ax10.plot(times_b4,np.mean(ACF_b4,axis=0),color='k', lw=0.7)
ax10.plot(times_b4,np.mean(ACF_fit_b4,axis=0),color='magenta')

point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('B4',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)


ax11 = fig.add_subplot(gs4[1,0],sharex=ax10) # 2D ACF B4                                               
ax11.imshow(ACF_b4,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times_b4[0],times_b4[-1],freqs_b4[0],freqs_b4[-1]))
T,F=np.meshgrid(times_b4, freqs_b4)
ax11.contour(T,F,ACF_fit_b4,4, colors='magenta', linewidths=.5)
plt.setp(ax11.get_yticklabels(), visible=False)
ax11.set_xlabel('Time Lag [ms]')
point = ax11.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax11.legend((point,point), (' i',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 0.99), handletextpad=-0.5,frameon=True,markerscale=0)
plt.gca().add_artist(plotlabel)

ax12 = fig.add_subplot(gs4[1,1],sharey=ax11) #Freq ACF B4                                                    
ax12.plot(np.mean(ACF_b4,axis=1),freqs_b4,color='k', lw=0.7)
plt.setp(ax12.get_yticklabels(), visible=False)
plt.setp(ax12.get_xticklabels(), visible=False)
ax12.set_ylim(freqs_b4[0],freqs_b4[-1])
ax12.plot(np.mean(ACF_fit_b4,axis=1),freqs_b4,color='magenta')
point = ax12.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('n',), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax13  = fig.add_subplot(gs5[0,0]) # Time ACF B5
plt.setp(ax13.get_yticklabels(), visible=False)
plt.setp(ax13.get_xticklabels(), visible=False)
ax13.plot(times_b5,np.mean(ACF_b5,axis=0),color='k', lw=0.7)
ax13.plot(times_b5,np.mean(ACF_fit_b5,axis=0),color='skyblue')

point = ax13.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax13.legend((point,point), ('B5',), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.98), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
point = ax13.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax13.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

ax14 = fig.add_subplot(gs5[1,0],sharex=ax13) # 2D ACF B5
ax14.imshow(ACF_b5,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,extent=(times_b5[0],times_b5[-1],freqs_b5[0],freqs_b5[-1]))
T,F=np.meshgrid(times_b5, freqs_b5)
ax14.contour(T,F,ACF_fit_b5,4, colors='skyblue', linewidths=.5)
ax14.set_ylabel('Frequency Lag [MHz]')
ax14.set_xlabel('Time Lag [ms]')
point = ax14.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax14.legend((point,point), (' j',), loc='upper left',handlelength=0,bbox_to_anchor=(0.92, 0.99), handletextpad=-0.5,frameon=True,markerscale=0)
plt.gca().add_artist(plotlabel)

ax15 = fig.add_subplot(gs5[1,1],sharey=ax14) #Freq ACF B5
ax15.plot(np.mean(ACF_b5,axis=1),freqs_b5,color='k', lw=0.7)
plt.setp(ax15.get_yticklabels(), visible=False)
plt.setp(ax15.get_xticklabels(), visible=False)
ax15.set_ylim(freqs_b5[0],freqs_b5[-1])
ax15.plot(np.mean(ACF_fit_b5,axis=1),freqs_b5,color='skyblue')
point = ax15.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax15.legend((point,point), ('o',), loc='upper left',handlelength=0,bbox_to_anchor=(0.8, 0.99), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)

plt.savefig('edfigure2_KN.eps',format='eps',dpi=300)
plt.show()



