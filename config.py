import numpy as np



lensing_bins=16
source_bins=10



max_l=5000

redshift_binsize=(1-0.2)/lensing_bins



minzs=np.linspace(0.2,1-redshift_binsize,lensing_bins)


maxzs=np.linspace(0.2+redshift_binsize,1,lensing_bins)

lens_boundaries=np.linspace(0.2,1,lensing_bins+1)

#upperlimits=[2*np.pi*]#[ 420, 714, 930 ,1212 ]


'''
upperlimits=[359,
 443,
 525,
 604,
 682,
 757,
 830,
 901,
 970,
 1037,
 1101,
 1164,
 1225,
 1285,
 1342,
 1398]
'''
#biases=[1.35,1.5,1.65,1.8]#
'''
#biases=[1.35,
 1.38,
 1.42,
 1.46,
 1.5,
 1.53,
 1.57,
 1.61,
 1.65,
 1.69,
 1.74,
 1.78,
 1.82,
 1.86,
 1.9,
 1.95]#[1.35,1.5,1.65,1.8] #[0.95/power_spectra.growth(power_spectra.comoving_distance(centre-0.05))for centre in centres]

'''
