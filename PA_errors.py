import numpy as np
from scipy import special
from scipy.integrate import quad
import matplotlib.pyplot as plt

def PApdffunc(PA,PAtrue,Ltrue,sigmaI):
    """
    For integrating
    """
    P0 = Ltrue/sigmaI
    eta = P0*np.cos(2*(PA-PAtrue))/np.sqrt(2)

    a = (1/np.sqrt(np.pi))
    c = eta*np.exp(eta**2)*(1+special.erf(eta))
    d = np.exp(-(P0**2/2.))

    G = a*(a+c)*d
    return G

def PApdf(PAtrue, Ltrue, sigmaI,thresh=None):
    """
    Determines the probability distribution of the position angle around the true value
    Following Everett & Weisberg 2001

    """
    if thresh == None:
        thresh = 1.57
    if thresh < 1.57:
        thresh = 1.57

    PAs = np.linspace(-np.pi/2.,np.pi/2.,1000)

    P0 = Ltrue/sigmaI
    if P0 > thresh and P0<10:
        eta = P0*np.cos(2*((PAs-PAtrue)))/np.sqrt(2)

        a = (1/np.sqrt(np.pi))
        c = eta*np.exp(eta**2)*(1+special.erf(eta))
        d = np.exp(-(P0**2/2.))

        G = a*(a+c)*d
    if P0>=10:
        sig = 28.65*np.pi/180. * (1/P0)
        G = (1/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*((PAs-PAtrue)/sig)**2)
    if P0 <=thresh:
        G = np.zeros_like(PAs)
    return G

def PAerr(Ltrue, sigmaI):
    """
    Determines the ballpark error by numerically integrating under G from PApdf to encapsulate 68.28% under the pdf
    """
    hundred, err = quad(PApdffunc,-np.pi/2.,np.pi/2., args=(0, Ltrue, sigmaI))

    aim = (68.26/100.)*hundred

    lim = 6*np.pi/180.
    for i in range(2000):
        new, err = quad(PApdffunc,-lim,lim,args=(0, Ltrue, sigmaI))

        if new - aim == 0:
            print("final limit", lim*180./np.pi)
            print("percentage of pdf (aim 68.26%)", new/hundred*100)
            break
        if new - aim >0:
            if (new-aim) > 0.07:
                lim -= 0.01*np.pi/180.
            if (new-aim) <= 0.07:
                lim -=0.001*np.pi/180.
        if new - aim <0:
            if (aim-new)> 0.07:
                lim +=0.01*np.pi/180.
            if (aim-new) <= 0.07:
                lim +=0.001*np.pi/180.

    if i == 1999:
        print(i)
        print("final limit", lim*180./np.pi)
        print("percentage of pdf (aim 68.26%)", new/hundred*100)

    return lim*180./np.pi, new/hundred*100


"""
SNs=(np.arange(3.0,10.0,0.01))
lims = []
percents = []
for i in range(len(SNs)):

    lim,perc =  PAerr(SNs[i], 1.)
    lims.append(lim)
    percents.append(perc)

percents = np.array(percents)
print(np.max(np.abs(percents-68.26)))
lims = np.array(lims)
np.save("lowSN_PAerror.npy",lims)
"""
