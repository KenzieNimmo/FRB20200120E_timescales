"""
Script to make ED Figure 3 in Nimmo et al. 2021b

Kenzie Nimmo & Mark Snelders 2021

Data you need:
- NE2001 scattering map data
     - ne2001_dmfact99.9_highres.hdf5
- 2D ACF of the dynamic spectra of each burst
    - ACF_b*_8us_f8.npy
- The 2D Gaussian fit to the ACF, with the Lorentzian fit to the central component
    - fitACF_b*_8us_f8.npy

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord  
import astropy.units as u
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'

# set up M81R
# "ra 09h57m54.799s, dec +68d49m02.989s"
m81r = SkyCoord("09h57m54.799s +68d49m02.989s", unit=(u.hourangle, u.deg))

#set up R3
r3 = SkyCoord("01h58m00.75017s +65d43m00.3152", unit=(u.hourangle, u.deg))

# load in results from NE2001
df_all = pd.read_hdf("./data_products/ne2001_dmfact99.9_highres.hdf5")

# cut out the parts we want to plot (I originally let NE2001 run from l = -180 to l = 360 but it just wraps around after l = 180)
bmin, bmax = -90, 90
lmin, lmax = -180, 180
df =  df_all[(df_all["b"] >= bmin) &\
                (df_all["b"] <= bmax) &\
                (df_all["l"] >= lmin) &\
                (df_all["l"] <= lmax)].copy()

# extract the TAU, l and b from the df and put it in a 2d shape
heatmap_data_tau = df.pivot_table("TAU", "b", "l").copy()
heatmap_data_tau = heatmap_data_tau.iloc[::-1]

base_freq = 1.0 # GHz, output from NE2001
obs_freq = 1.4 # GHz
scale = -4.0 # frequency scaling for scattering timescale, note some people use -4.4

# lets use units of nanoseconds
ns_in_ms = 1000000.

# scale to ns and to 1.4 GHz
heatmap_data_tau_obs_freq_ns = heatmap_data_tau.multiply((obs_freq / base_freq)**scale).multiply(ns_in_ms)
temp_min = np.min(heatmap_data_tau_obs_freq_ns.to_numpy())
temp_max = np.max(heatmap_data_tau_obs_freq_ns.to_numpy())

lon = np.linspace(np.pi, -np.pi, heatmap_data_tau_obs_freq_ns.shape[1])
lat = np.linspace(np.pi / 2., -np.pi / 2., heatmap_data_tau_obs_freq_ns.shape[0])
Lon,Lat = np.meshgrid(lon, lat)

# load in ACFs and fits                                                                                
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
fig = plt.figure(figsize=(8,8))
rows=5
cols=1

gs = gridspec.GridSpec(ncols=cols, nrows=rows,top=0.99,bottom=0.64, wspace=0.0, hspace=0)
gs2 = gridspec.GridSpec(ncols=1, nrows=1,top=0.56,bottom=0.05,wspace=0.0)

ax1 = fig.add_subplot(gs[0,0])
ax1.text(-0.1,0.9,"a",transform=ax1.transAxes)
point = ax1.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('B1',), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
resid1 = np.sum(ACF_b1,axis=1)-np.sum(ACF_fit_b1,axis=1)
resid1 = np.ma.masked_where(resid1==np.max(resid1),resid1)
ax1.plot(freqs_b1,resid1/np.max(resid1),color='k')
ax1.plot(freqs_b1,lor_b1/np.max(resid1),color='orange',label=r'$1.7\pm0.6$ MHz')
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_xlim(-150,150)
ax1.legend(loc='upper right')
ax1.set_yticks([0,1])

ax2 = fig.add_subplot(gs[1,0],sharex=ax1)
point = ax2.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('B2',), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
resid2 = np.sum(ACF_b2,axis=1)-np.sum(ACF_fit_b2,axis=1)
resid2 = np.ma.masked_where(resid2==np.max(resid2),resid2)
ax2.plot(freqs_b2,resid2/np.max(resid2),color='k')
ax2.plot(freqs_b2,lor_b2/np.max(resid2),color='purple',label=r'$3.0\pm0.8$ MHz')
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_xlim(-150,150)
ax2.legend(loc='upper right')
ax2.set_yticks([0,1])

ax3 = fig.add_subplot(gs[2,0],sharex=ax2)
point = ax3.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('B3',), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
resid3 = np.sum(ACF_b3,axis=1)-np.sum(ACF_fit_b3,axis=1)
resid3 = np.ma.masked_where(resid3==np.max(resid3),resid3)
ax3.plot(freqs_b3,resid3/np.max(resid3),color='k')
ax3.plot(freqs_b3,lor_b3/np.max(resid3),color='green',label=r'$5.8\pm0.7$ MHz')
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_xlim(-150,150)
ax3.legend(loc='upper right')
ax3.set_yticks([0,1])

ax4 = fig.add_subplot(gs[3,0],sharex=ax3)
point = ax4.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B4',), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
resid4 = np.sum(ACF_b4,axis=1)-np.sum(ACF_fit_b4,axis=1)
resid4 = np.ma.masked_where(resid4==np.max(resid4),resid4)
ax4.plot(freqs_b4,resid4/np.max(resid4),color='k')
ax4.plot(freqs_b4,lor_b4/np.max(resid4),color='magenta',label=r'$5.4\pm0.8$ MHz')
plt.setp(ax4.get_xticklabels(), visible=False)
ax4.set_xlim(-150,150)
ax4.legend(loc='upper right')
ax4.set_yticks([0,1])

ax5 = fig.add_subplot(gs[4,0],sharex=ax4)
ax5.text(-0.1,-1.5,"b",transform=ax5.transAxes)
point = ax5.scatter(0,0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('B5',), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
resid5 = np.sum(ACF_b5,axis=1)-np.sum(ACF_fit_b5,axis=1)
resid5 = np.ma.masked_where(resid5==np.max(resid5),resid5)
ax5.plot(freqs_b5,resid5/np.max(resid5),color='k')
ax5.plot(freqs_b5,lor_b5/np.max(resid5),color='skyblue',label=r'$2.9\pm1.5$ MHz')
ax5.set_xlim(-150,150)
ax5.set_xlabel('Frequency Lag [MHz]')
ax5.legend(loc='upper right')
ax5.set_yticks([0,1])

fig.text(0.04,0.82,'Normalised ACF',va='center', rotation='vertical')

ax = fig.add_subplot(gs2[0,0], projection='mollweide')
im = ax.pcolormesh(Lon,Lat,heatmap_data_tau_obs_freq_ns, cmap='rainbow',\
    shading='auto',\
    norm=colors.LogNorm(vmin=temp_min, vmax=temp_max / 1e11)) # saturate the plot to show more detail at abs(b) > 30

cbar = fig.colorbar(im, ax=ax, extend='max',orientation="horizontal")
cbar.ax.set_xlabel(r"Scattering timescale [ns]")

fig.text(0.08,0.43,'Galactic Latitude',rotation='vertical')
fig.text(0.45,0.35,'Galactic Longitude')
ax.set_xticks([-150.*np.pi/180., -120.*np.pi/180., -90.*np.pi/180., -60.*np.pi/180., -30.*np.pi/180., 0,\
               30.*np.pi/180., 60.*np.pi/180., 90.*np.pi/180., 120.*np.pi/180., 150.*np.pi/180.])
ax.set_xticklabels([r'150$^{\circ}$', r'120$^{\circ}$', r'90$^{\circ}$', r'60$^{\circ}$', r'30$^{\circ}$',\
                    r'0$^{\circ}$', r'-30$^{\circ}$', r'-60$^{\circ}$', r'-90$^{\circ}$',\
                    r'-120$^{\circ}$', r'-150$^{\circ}$'],\
                   color='k')

# put yticks in the same style
ax.set_yticks([75.*np.pi/180., 60.*np.pi/180., 45.*np.pi/180., 30.*np.pi/180., 15.*np.pi/180., 0,\
               -15.*np.pi/180., -30.*np.pi/180., -45.*np.pi/180., -60.*np.pi/180., -75.*np.pi/180.])
ax.set_yticklabels([r'75$^{\circ}$', r'60$^{\circ}$', r'45$^{\circ}$', r'30$^{\circ}$', r'15$^{\circ}$',\
                    r'0$^{\circ}$', r'-15$^{\circ}$', r'-30$^{\circ}$', r'-45$^{\circ}$',\
                    r'-60$^{\circ}$', r'-75$^{\circ}$'],\
                   color='k')

# place crosses at M81R and R3
ax.scatter([-m81r.galactic.l.value * np.pi / 180.], [m81r.galactic.b.value * np.pi / 180.], s=50,\
          color='black', marker='x', linewidths=1.)
fig.text(0.28,0.45,'FRB 20200120E',color='black')
ax.scatter([-r3.galactic.l.value * np.pi / 180.], [r3.galactic.b.value * np.pi / 180.], s=50,\
          color='white', marker='+', linewidths=1.)
fig.text(0.22,0.36,'FRB 20180916B',color='white')

plt.grid(True, alpha=0.5, color='white')
plt.savefig('edfigure3_KN.eps',format='eps',dpi=300)     
plt.show()
