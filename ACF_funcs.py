import numpy as np
from scipy.fftpack import fft, ifft, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from time import sleep
from tqdm import tqdm
from scipy.optimize import curve_fit


def shift(v, i, nchan):
        """                                                                                                                                                            
        function v by a shift i                                                                                                                                        
        nchan is the number of frequency channels (to account for negative lag)                                                                                        
        """
        n = len(v)
        r = np.zeros(3*n)
        i+=nchan-1 #to account for negative lag                                                                                                                        
        i = int(i)
        r[i:i+n] = v
        return r

def autocorr(x, nchan, v=None,zerolag=False):
    if v is None:
        v = np.ones_like(x)
    x = x.copy()
    x[v!=0] -= x[v!=0].mean()
    ACF = np.zeros_like(x)
    for i in tqdm(range(len(x))):
        if zerolag == False:
                if i>0:
                        m = shift(v,0,nchan)*shift(v,i,nchan)
                        ACF[i-1] = np.sum(shift(x,0,nchan)*shift(x, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(x, i, nchan)**2*m))
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.sum(shift(x,0,nchan)*shift(x, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(x, i, nchan)**2*m))
            

    return ACF


def skewness(x,nchan):
        x = x.copy()
        x2= x**2
        x3 = x**3
        skew = np.zeros_like(x)
        for i in tqdm(range(len(x))):
                skew[i] = (1/(len(x)-i))*(np.sum(shift(x2,0,nchan)[:(len(x)-i)]*shift(x, i,nchan)[:(len(x)-i)])-np.sum(shift(x,0,nchan)[:(len(x)-i)]*shift(x2, i,nchan)[:(len(x)-i)]))/np.mean(x3)

        return skew

def corr(x,y, nchan, v=None,zerolag=False):
    if v is None:
        v = np.ones_like(x)
    x = x.copy()
    y = y.copy()
    x[v!=0] -= x[v!=0].mean()
    y[v!=0] -= y[v!=0].mean()
    ACF = np.zeros_like(x)
    for i in tqdm(range(len(x))):
        if zerolag == False:
                if i>0:
                        m = shift(v,0,nchan)*shift(v,i,nchan)
                        ACF[i-1] = np.sum(shift(x,0,nchan)*shift(y, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(y, i, nchan)**2*m))
        else:
                m = shift(v,0,nchan)*shift(v,i,nchan)
                ACF[i] = np.sum(shift(x,0,nchan)*shift(y, i,nchan)*m)/np.sqrt(np.sum(shift(x, 0, nchan)**2*m)*np.sum(shift(y, i, nchan)**2*m))


    return ACF

def lorentz(x,gamma,y0, c):
        return (y0*gamma)/(((x)**2)+gamma**2)+c

def gaus(x,a,x0,sigma,c):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def autocorr_fft(x):
    xp = ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    #return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)                                                                                                            
    return np.real(pi)[:n]/(np.arange(n)[::-1]+n)

def autocorr_np(x):
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')/xp.size
    #return result[result.size/2:]/len(xp)                                                                                                                             
    return result[result.size/2:]

def line(x, B): 
    return  0*x+B

def doublegaus(x,m1,m2,s1,s2,a1,a2,o1,o2):
        return gaus(x,a1,m1,s1,o1)+gaus(x,a2,m2,s2,o2)


def crosscoef(ds,prof,SN,dsoff,returnvals=False):
    """
    Given a dynamic spectrum of a burst (converted to S/N), compute the cross coefficient of each bin spectra with every other time bin
    Plots the cross correlation coefficient as a function of time lag between the two spectra

    Inputs:
         - ds = dynamic spectrum
         - prof = profile in S/N
         - SN = Signal to noise threshold 
         - noisespec = spectrum of off burst noise
    """
    indices=np.where(prof>SN)[0]
    
    corrcoefs=np.zeros((len(indices),len(indices)))
    positions=np.zeros((len(indices),len(indices)))
    geometric_means = np.zeros((len(indices),len(indices)))
    for i in range(len(indices)):
        for j in range(len(indices)):
             
             Stokes1=ds[:,indices[i]]-np.mean(dsoff,axis=1)#-noisespec#-np.mean(noisespec)
             Stokes2=ds[:,indices[j]]-np.mean(dsoff,axis=1)
             
             #newnoisespec = noisespec-np.mean(noisespec)
             #Stokes1/=np.std(newnoisespec)
             #Stokes2/=np.std(newnoisespec)
             corrcoefs[i,j]=np.corrcoef(Stokes1,Stokes2)[0,1]
             positions[i,j]=np.abs(indices[j]-indices[i])*1
             geometric_means[i,j]=np.sqrt(prof[indices[j]]*prof[indices[i]])


    #get unique values of time lags in order to determine the weighted mean of the corr coeffs per delta time
    timelags=np.unique(positions)
    timelags = timelags[np.where(timelags!=0)[0]]

    corrcoefs=corrcoefs[np.where(positions!=0)]
    geometric_means=geometric_means[np.where(positions!=0)]
    positions = positions[np.where(positions!=0)]

    corrcoefs_mean = np.zeros_like(timelags)
    corrcoefs_std = np.zeros_like(timelags)
    SNscale = geometric_means/np.max(geometric_means)

    for i in range(len(timelags)):
        indices_lag = np.where(positions==timelags[i])
        corrcoefs_mean[i]=np.sum(SNscale[indices_lag]*corrcoefs[indices_lag])/np.sum(SNscale[indices_lag])#np.average(corrcoefs[indices_lag],weights=SNscale[indices_lag])
        n = len(SNscale[indices_lag])
        numerator = np.sum(n * SNscale[indices_lag] * (corrcoefs[indices_lag] - corrcoefs_mean[i]) ** 2.0)
        denominator = (n - 1) * np.sum(SNscale[indices_lag])
        weighted_std = np.sqrt(numerator / denominator)
        corrcoefs_std[i]=weighted_std
        

    #try fit a horizontal line to the data (weight using S/N)
    popt, pcov = curve_fit(line, positions, corrcoefs, p0=[0.3], sigma=1/SNscale)
    print("Line is:", popt[0],"+-",pcov[0][0])

    fig = plt.figure() #Correlation coefficient
    im=plt.scatter(positions,corrcoefs,marker='x',c=geometric_means,cmap='Purples',alpha=1.0)
    #plt.errorbar(timelags,corrcoefs_mean,yerr=corrcoefs_std,color='k',alpha=0.5)
    plt.plot(timelags,line(timelags,*popt),'k')
    #plt.ylim(-0.2,1)

    #plt.set_xlim(0,65)
    plt.ylabel('Correlation coefficient')
    plt.xlabel(u'Time lag')
    cbar=fig.colorbar(im)
    cbar.set_label("Geometric mean S/N",rotation=-90,labelpad=20)
        
    if returnvals ==True:
            plt.close()
            return positions,corrcoefs,geometric_means,popt[0]
    else:
            return plt.show()


def autocorr_2D(arr):
    """
    Given an array arr, compute the ACF of the array
    """
    ACF= correlate2d(arr, arr)
    return ACF


