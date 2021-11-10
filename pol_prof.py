import numpy as np
from scipy.interpolate import interp1d
import PA_errors
import matplotlib.pyplot as plt 

def remove_delay(Q,U,freq,delay):
    """
    Remove the delay between pol hands from the data

    Inputs:
    - Q: dynamic spectrum Stokes Q
    - U: dynamic spectrum Stokes U
    - freq: array of the frequencies (in MHz)
    - cdelay: the delay in nanoseconds
    
    Outputs:
    - corrected Q and U
    """
    remove_delay=np.exp(-2*1j*(freq*1e6)*delay*1e-9*np.pi)

    Qcorr=np.zeros_like(Q)
    Ucorr=np.zeros_like(U)
    
    for f in range(Q.shape[1]):
        Qspec = Q[:,f]
        Uspec = U[:,f]
        lin=Qspec+1j*Uspec
        lin*=remove_delay
        Qcorr[:,f]=lin.real
        Ucorr[:,f]=lin.imag

    return Qcorr, Ucorr

def remove_RM(Q,U,freq,RM):
    """
    De-Faradays the data
    Input:
    - Q: dynamic spectrum Stokes Q                                                                                                                                   
    - U: dynamic spectrum Stokes U                                                                                                                                    
    - freq: array of the frequencies (in MHz)                                                                                                                         
    - RM: the rotation measure [rad/m^2]

    Outputs:
    - corrected Q and U 
    """
    remove_faraday=np.exp(-2*1j*((3e8)**2/(freq*1e6)**2)*(RM))
    Qcorr=np.zeros_like(Q)
    Ucorr=np.zeros_like(U)
    for f in range(Q.shape[1]):
        Qspec = Q[:,f]
        Uspec = U[:,f]
        lin=Qspec+1j*Uspec
        lin*=remove_faraday
        Qcorr[:,f]=lin.real
        Ucorr[:,f]=lin.imag

    return Qcorr, Ucorr

def centre_burst(I,Q,U,V,StokesI_ds,tcent=None):
    """
    roll burst profiles to put the burst in the centre 

    Input:
    - I, Q, U, V: time series of the four stokes parameters

    Output:
    - Centred I, Q, U, V
    """
    peakbin = np.argmax(I)
    if tcent!=None:
        peakbin=int(tcent)
    newpeak = int(len(I)/2.)
    if peakbin > newpeak:
        I = np.roll(I,(len(I)-peakbin)+newpeak)
        V = np.roll(V,(len(I)-peakbin)+newpeak)
        Q = np.roll(Q,(len(I)-peakbin)+newpeak)
        U = np.roll(U,(len(I)-peakbin)+newpeak)
        StokesI_ds = np.roll(StokesI_ds,(len(I)-peakbin)+newpeak,axis=1)
        offset=(len(I)-peakbin)+newpeak
    if peakbin < newpeak:
        I = np.roll(I,newpeak-peakbin)
        V = np.roll(V,newpeak-peakbin)
        Q = np.roll(Q,newpeak-peakbin)
        U = np.roll(U,newpeak-peakbin)
        StokesI_ds = np.roll(StokesI_ds,newpeak-peakbin,axis=1)
        offset=newpeak-peakbin
        
    while offset > len(I)/2.:
        offset-=len(I)

    return I, Q, U, V, StokesI_ds,offset
    
def PPA(U,Q,Lunbias,Ioffstd,parang,PAoffset=None):
    """
    Computes the PPA probability distribution per time bin    
    """
    PA = np.zeros_like(U)
    for i in range(len(U)):
        PA[i]=(0.5*np.arctan2(U[i],(Q[i])))
    PA*=(180./np.pi) #converting to degrees               
    PA+=parang #rotating the linear polarisation vector by the parallactic angle                                                                                      
    PA_notrot = PA.copy()
    if (PA>90).any():
        PA[PA>90]-=180
    if (PA<-90).any():
        PA[PA<-90]+=180
    if PAoffset!=None:
        PA+=PAoffset #rotating by some offset (for Nimmo et al. 2020 we use the weighted average of the four bursts)                                                  
        
    PAmask = np.ones_like(U)
    ind = np.where((Lunbias/Ioffstd>3)) #setting a sigma threshold of 3 (below which we don't plot the PPA)                                                           
    PAmask[ind] = 0
    PAnomask = PA
    PA = np.ma.masked_where(PAmask==True,PA)
    x = np.linspace(0,len(PA)+1,len(PA)) #an x array for the PPA                                                                                                       
    #errors on PPA                                                                                                                                                    
    #load in an array to interpolate to find the errors for low linear S/N (following Everett and Weisberg 2001)                                                      
    #for high S/N (anything above sigma = 10) use 28.65*sigmaI/unbiased_linear (see Everett and Weisberg 2001)                                                        
    lowSN = np.load("/home/nimmo/scripts/lowSN_PAerror.npy")
    SNs = np.arange(3.0,10.0,0.01)
    SNsfunc = interp1d(SNs,lowSN)
    PAerror=[]
    weights = []
    for i in range(len(Lunbias)):
        if Lunbias[i]/Ioffstd >= 10.:
            PAerror.append(28.65*Ioffstd/Lunbias[i])
            weights.append(1./(28.65*Ioffstd/Lunbias[i]))
        if Lunbias[i]/Ioffstd>=3 and Lunbias[i]/Ioffstd < 10.:
            PAerror.append(SNsfunc(Lunbias[i]/Ioffstd))
            weights.append(1./(SNsfunc(Lunbias[i]/Ioffstd)**2))
        if Lunbias[i]/Ioffstd<3.:
            PAerror.append(0)
            weights.append(0)

    #pdf of PA (following Everett and Weisberg 2001)                                                                                                                 
    pdf = []
    for i in range(len(Lunbias)):
        pdf.append(PA_errors.PApdf(PAnomask[i]*np.pi/180., Lunbias[i], Ioffstd,thresh=3))
    pdf = np.array(pdf)
    pdf = pdf.T

    return pdf,x,PA,weights, PAerror,PA_notrot

def PPA_fit(x,PA,weights):
    """
    """

    rot=False
    fit=np.polyfit(x,PA,0,w=weights,full=True)
    PAmean=fit[0][0]
    burstweights=np.array(weights)
    burstPAs = np.array(PA)
    dof = len(np.where(burstweights!=0)[0])-1
    print('dof', dof)
    #calculate chisq                                                                                                                                                  
    chisq_k = np.sum((burstPAs-PAmean)**2*burstweights)
    redchisq=chisq_k/dof
    print("chisq",chisq_k)
    print("reduced chisq",redchisq)
    print("mean offset PPA",PAmean)
    
    if (PAmean>90):
        PAmean-=180
    if (PAmean<-90):
        PAmean+=180
    return PAmean


def get_profile(ds,freqs,delay,RM,parang,burstbeg=None, burstend=None, tcent=None, PAoffset=None, name=None):

    """
    ds is with dimensions polarisation, channel, profile bin.
    freqs is an array of frequencies
    delay is the delay between pol hands in nanoseconds
    RM is the rotation measure
    parang is the parallactic angle in degrees
    Polarisation is in the form RR, LL, RL_real, RL_imaginary

    Optional arguments: PAoffset to shift the polarisation position angle by some angle
                        (for this paper (Nimmo et al. 2020) we use the weighted mean PPA of the four bursts).
                        burstbeg burstend are the beginning and end times of the burst to compute the polarisation fractions
                        tcent is the centre time (default to use the peak time, in bins) of the burst (to put at time=0 -- matches what we did previously),
                        if not given, the peak of the burst is taken as tcent.
                        name is for saving numpy arrays with the filename name_*.npy. If name=None, no files will be saved
                        window to define the +- window in seconds around the plot to output the profiles, if None it outputs all the data
                        window_for_off to define the window around the burst to exclude from off burst statistics (default 5ms)
    
    Stokes params are defined as (PSR/IEEE convention)
    I = RR + LL
    Q = 2*RL_real
    U = 2*i*RL_imaginary
    V = LL - RR

    linear is sqrt(Q^2+U^2) which is equivalent to sqrt(Q^2-abs(U^2))
    """
    #read in data
    RR = ds[0,:,:]
    LL = ds[1,:,:]
    RLr = ds[2,:,:]
    RLi = ds[3,:,:]
    
    #remove the polarisation delay
    RLr,RLi=remove_delay(RLr,RLi,freqs,delay)
    RLr,RLi=remove_RM(RLr,RLi,freqs,RM)
    
    StokesI_ds = RR+LL

    if burstbeg == None and burstend == None:
        plt.plot(np.mean(RR+LL,axis=0))
        plt.show()
        answer = input("Begin and end bin of the burst (begin,end) ")
        burstbeg = answer[0]
        burstend = answer[1]

    b = int(burstbeg-50)
    e = int(burstend+50)

    if b < 0:
        offRR = RR[:,e:]
        offLL = LL[:,e:]
        offRLr = RLr[:,e:]
        offRLi = RLi[:,e:]
    elif e > RR.shape[1]:
        offRR = RR[:,:b]
        offLL = LL[:,:b]
        offRLr = RLr[:,:b]
        offRLi = RLi[:,:b]
    else:
        offRR = np.concatenate((RR[:,:b],RR[:,e:]),axis=1)
        offLL = np.concatenate((LL[:,:b],LL[:,e:]),axis=1)
        offRLr = np.concatenate((RLr[:,:b],RLr[:,e:]),axis=1)
        offRLi = np.concatenate((RLi[:,:b],RLi[:,e:]),axis=1)

    offspec = np.mean(offRR+offLL,axis=1)
    
    RRprofile = np.sum(RR,axis=0)
    RRoffprof = np.sum(offRR,axis=0)

    LLprofile = np.sum(LL,axis=0)
    LLoffprof = np.sum(offLL,axis=0)

    RLrprofile = np.sum(RLr,axis=0)
    RLroffprof = np.sum(offRLr,axis=0)

    RLiprofile = np.sum(RLi,axis=0)
    RLioffprof = np.sum(offRLi,axis=0)

    #create Stokes parameters with correct statistics
    I=RRprofile+LLprofile
    Q = 2*RLrprofile
    U = 2*RLiprofile
    V=RRprofile-LLprofile

    
    Ioff = RRoffprof+LLoffprof
    Qoff = 2*RLroffprof
    Uoff = 2*RLioffprof
    Voff = RRoffprof-LLoffprof

    #statistics should be ~ the same for all four Stokes parameters
    #use 1% as limit within which the statistics should agree
    Ivar=np.var(Ioff)
    Ivar_var = 0.01*Ivar
    sigmaQ = np.std(Qoff)
    sigmaU = np.std(Uoff)
    sigmaV = np.std(Voff)
    Ioffstd = np.std(Ioff)

    if ((Ivar+Ivar_var)>np.var(Qoff)>(Ivar-Ivar_var)) and ((Ivar+Ivar_var)>np.var(Uoff)>(Ivar-Ivar_var)) and ((Ivar+Ivar_var)>np.var(Voff)>(Ivar-Ivar_var)):
        print("Ivar", np.std(Ioff))
        print("Qvar", np.std(Qoff))
        print("Uvar", np.std(Uoff))
        print("Vvar", np.std(Voff))
    
    else:
        #StokesI_ds/=np.std(offRR+offLL)
        I/=Ioffstd
        Ioff/=Ioffstd
        Q/=sigmaQ
        Qoff/=sigmaQ
        U/=sigmaU
        Uoff/=sigmaU
        V/=sigmaV
        Voff/=sigmaV
        #Note we remove the baseline already in the load_archive function
        print("Ivar", np.var(Ioff))
        print("Qvar", np.var(Qoff))
        print("Uvar", np.var(Uoff))
        print("Vvar", np.var(Voff))
        sigmaQ = np.std(Qoff)
        sigmaU = np.std(Uoff)
        sigmaV = np.std(Voff)
        Ioffstd = np.std(Ioff)
        
    Ioffstd = np.std(Ioff)
    print("Ioffstd", Ioffstd)

    I,Q,U,V,StokesI_ds,offset = centre_burst(I,Q,U,V,StokesI_ds,tcent)
    burstbeg +=offset
    burstend+=offset

    linear = np.sqrt(Q**2+U**2)
    

    #debiasing following Everett and Weisberg 2001
    Ltrue = np.zeros_like(linear)
    for i in range(len(linear)):
        if linear[i]/Ioffstd > 1.57:
            Ltrue[i] = Ioffstd*np.sqrt((linear[i]/Ioffstd)**2-1)

    #errors on Ltrue
    sigmaLtrue = np.zeros_like(Ltrue)
    for i in range(len(Ltrue)):
        if Ltrue[i]>0:
            sigmaL = 0.5*np.sqrt(2*(Q[i]*sigmaQ)**2+2*(U[i]*sigmaU)**2)*(1/linear[i])
            sigmaLtrue2 = 2*sigmaL*Ltrue[i]**2/linear[i]
            sigmaLtrue[i] = 0.5*sigmaLtrue2/Ltrue[i]

    #polarisation position angle (PPA)
    pdf,x,PA,weights,PAerror,PA_notrot = PPA(U,Q,Ltrue,Ioffstd,parang,PAoffset=PAoffset)
    x = np.arange(0,len(PA),1)
    x -= int(len(x)/2.)
    
    #fit a straight horizontal line to PA using the correct errors
    # to test the hypothesis that the PPA is flat across the band
    burstbeg = int(burstbeg)
    burstend = int(burstend)
    PAmean = PPA_fit(x[burstbeg:burstend],PA_notrot[burstbeg:burstend],weights[burstbeg:burstend])

    totL=np.sum(Ltrue[burstbeg:burstend])
    #errors on fractional polarisation
    sigmaLtot=np.sqrt(np.sum(np.array(sigmaLtrue[burstbeg:burstend])**2))
    sigmaItot= np.sqrt(np.sum((np.zeros_like(sigmaLtrue[burstbeg:burstend])+Ioffstd)**2))
    sigmaVtot= np.sqrt(np.sum((np.zeros_like(sigmaLtrue[burstbeg:burstend])+sigmaV)**2))

    totI=np.sum(I[burstbeg:burstend])
    sigmaLI = totL/totI*np.sqrt((sigmaLtot/totL)**2+(sigmaItot/totI)**2)

    totVabs=np.sum(np.abs(V[burstbeg:burstend]))
    totV=np.sum(V[burstbeg:burstend])
    sigmaVI = totV/totI*np.sqrt((sigmaVtot/totV)**2+(sigmaItot/totI)**2)
    print("linear frac (L/I)",totL/totI,"+-",sigmaLI)
    print("abs V frac (V/I)",totVabs/totI)
    print("circular frac (V/I)", totV/totI,"+-",sigmaVI)

    if name!=None:
        np.save('%s_I.npy'%name,I)
        np.save('%s_V.npy'%name,V)
        np.save('%s_linear.npy'%name,Ltrue)

    x = np.array(map(float, x))

    return StokesI_ds, I, Ltrue, V, PA, x, PAmean, PAerror, pdf, weights,burstbeg,burstend
    


def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
