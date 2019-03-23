import numpy as np



lensing_bins=4
source_bins=10



max_l=5000

redshift_binsize=(1-0.2)/lensing_bins



minzs=np.linspace(0.2,1-redshift_binsize,lensing_bins)


maxzs=np.linspace(0.2+redshift_binsize,1,lensing_bins)

lens_boundaries=np.linspace(0.2,1,lensing_bins+1)


