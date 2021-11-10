from astropy import units as u
import numpy as np
from astropy.coordinates import Angle


def parangle(MJD,longitude,RA,dec,lat,hang=None):
    """
    MJD is a float -- the modified julian date of the burst
    longitude is a float -- the longitude of the telescope
    RA is an Angle -- right ascension of the source
    dec is an Angle -- declination of the source
    lat is an Angle -- latitude of the observatory
    """
    LST = LSTfromMJD(MJD,longitude)
    if hang!=None:
        ha_rad=hang
    else:
        ha = LST-RA
        if ha < Angle('0h'):
            ha=Angle('24h')+ha
        if ha>Angle('12h'):
            ha=ha-Angle('24h')


        ha+=Angle('24h')


        ha_rad = ha.to(u.radian).value
    dec_rad = dec.to(u.radian).value
    lat_rad = lat.to(u.radian).value
    pa = np.arctan2(np.sin(ha_rad)*np.cos(lat_rad),(np.cos(dec_rad)*np.sin(lat_rad)-np.cos(lat_rad)*np.sin(dec_rad)*np.cos(ha_rad)))

    return pa*180/np.pi

def LSTfromMJD(MJD,lon):
    """
    Convert MJD to LST Angle 
    lon = longitude in degrees
    """
    JD = MJD + 2400000.5
    #calculate the Greenwhich mean sidereal time:
    GMST = 18.697374558 + 24.06570982441908*(JD - 2451545)
    GMST = GMST % 24    #use modulo operator to convert to 24 hours
    GMSTmm = (GMST - int(GMST))*60          #convert fraction hours to minutes
    GMSTss = (GMSTmm - int(GMSTmm))*60      #convert fractional minutes to seconds
    GMSThh = int(GMST)
    GMSTmm = int(GMSTmm)
    GMSTss = int(GMSTss)


    #Convert to the local sidereal time by adding the longitude (in hours) from the GMST.
    #(Hours = Degrees/15, Degrees = Hours*15)
    Long = lon/15      #Convert longitude to hours
    LST = GMST+Long     #Fraction LST. If negative we want to add 24...
    if LST < 0:
        LST = LST +24
    LSTmm = (LST - int(LST))*60          #convert fraction hours to minutes
    LSTss = (LSTmm - int(LSTmm))*60      #convert fractional minutes to seconds
    LSThh = int(LST)
    LSTmm = int(LSTmm)
    LSTss = int(LSTss)
    
    print '\nLocal Sidereal Time %s:%s:%s \n\n' %(LSThh, LSTmm, LSTss)
    return Angle(str(LSThh)+'h'+str(LSTmm)+'m'+str(LSTss)+'s')
