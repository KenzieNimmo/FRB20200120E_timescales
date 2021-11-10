import psrchive
import numpy as np
import filterbank


def load_archive(archive_name,rm=0,dm=0,tscrunch=None,fscrunch=None,remove_baseline=True,extent=False,model=False,pscrunch=False,cent=False):
    """
    Takes an archive file with name, archive_name, and converts to a numpy array with masks
    """

    print("Loading in archive file {}".format(archive_name))
    archive = psrchive.Archive_load(archive_name)
    if cent==True:
        archive.centre()

    archive.tscrunch()
    if pscrunch == True:
        archive.pscrunch()

    ardm =archive.get_dispersion_measure()
    ardd = archive.get_dedispersed()
    arfc = archive.get_faraday_corrected()
    arrm = archive.get_rotation_measure()


    if ardd==True:
        print("Archive file is already dedispersed to a DM of {} pc/cc".format(ardm))
        dm = dm - ardm
        


    if remove_baseline == True:
        #remove the unpolarised background -- note this step can cause Stokes I < Linear
        archive.remove_baseline()

    if tscrunch!=None and tscrunch!=1:
        archive.bscrunch(tscrunch)

    if fscrunch!=None and fscrunch!=1:
        archive.fscrunch(fscrunch)

    if dm!=0:
        if ardd==True and dm==0:
            pass
        if ardd==True and ardm!=dm:
            print("Dedispersing the data to a DM of {} pc/cc".format(dm+ardm))
        else: print("Dedispersing the data to a DM of {} pc/cc".format(dm))
        archive.set_dispersion_measure(dm)
        archive.set_dedispersed(False)
        archive.dedisperse()

    if rm!=0:
        if arfc==True:
            print("Faraday derotating the data to a RM of {} rad/m^2".format(rm+arrm))
        else: print("Faraday derotating the data to a RM of {} rad/m^2".format(rm))
        archive.set_rotation_measure(rm)
        archive.set_faraday_corrected(False)
        archive.defaraday()

    ds = archive.get_data().squeeze()
    w = archive.get_weights().squeeze()
    if model==False:
        if len(ds.shape)==3:
            for j in range(np.shape(ds)[0]):
                ds[j,:,:] = (w[:]*ds[j,:,:].T).T
            ds = np.flip(ds,axis=1)
        else: 
            ds = (w*ds.T).T
            ds = np.flip(ds,axis=0)
            
    
    if model==True:
        ds = ds

    tsamp = archive.get_first_Integration().get_duration()/archive.get_nbin()

    
    if extent==True:
        extent = [0, archive.get_first_Integration().get_duration()*1000,\
        archive.get_centre_frequency()+archive.get_bandwidth()/2.,\
        archive.get_centre_frequency()-archive.get_bandwidth()/2.]
        return ds, extent,tsamp
    else:
        return ds


def load_filterbank(filterbank_name,dm=None,fullpol=False):
    """
    Takes an filterbank file with name, filterbank_name, and converts to a numpy array 
    """

    fil = filterbank.FilterbankFile(filterbank_name)
    spec = fil.get_spectra(0,fil.nspec)
    if dm!=None and fullpol==True:
        raise ValueError("If filterbank contains full polarisation information, dedispersion won't work properly")
    if dm != None:
        spec.dedisperse(dm)
    
    arr=spec.data
    if fullpol==True:
        arr_reshape = arr.reshape(fil.header['nchans'],-1,4)
        arr = arr_reshape

    if fil.header['foff'] < 0:
        #this means the band is flipped
        arr = np.flip(arr,axis=0)
        foff = fil.header['foff']*-1
    else: foff = fil.header['foff']

    #header information
    tsamp = fil.header['tsamp']
    begintime = 0
    endtime = arr.shape[1]*tsamp
    fch_top = fil.header['fch1']
    nchans = fil.header['nchans']
    fch_bottom = fch_top+foff-(nchans*foff)

    extent = (begintime,endtime,fch_bottom,fch_top)

    return arr, extent, tsamp
