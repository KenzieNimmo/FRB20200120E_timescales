# FRB20200120E_timescales
A collection of scripts used for the analysis and plotting in Nimmo et al. 2021b "Burst timescales and luminosities as links between young pulsars and fast radio bursts" --> https://ui.adsabs.harvard.edu/abs/2021arXiv210511446N/abstract

The data required to run these scripts can be found here:


- Scripts titled "figure*.py" or "edfigure*.py" are used to make the corresponding figures in Nimmo et al. 2021b
- The .txt files contain the data required for the transient phase space diagram (Figure 3)
- ACF_funcs.py contains the autocorrelation functions to perform 1D and 2D ACFs, Lorentzian fits, and compute the correlation coefficients
- radiometer.py contains the radiometer equation for converting from S/N to flux units
- pol_prof.py contains functions to take a full pol archive file and output the dynamic spectra, full pol profiles and polarisation angle for plotting after correcting for instrumental delay and faraday rotation
- parallactic_angle.py contains functions to compute the parallactic angle correction for a burst (which will be corrected within pol_prof.py)
- load_file.py contains functions to input archive files or filterbank files and output dedispersed dynamic spectra
