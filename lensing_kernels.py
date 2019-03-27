#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:08:26 2019

@author: fionamccarthy
"""

import numpy as np

import cosmo_functions

import matplotlib.pyplot as plt

from scipy import integrate


from scipy import optimize


H0=cosmo_functions.H0
c=cosmo_functions.c
omegam=cosmo_functions.omegam

#lensing_bins=config.lensing_bins

source_bins=10


#redshift_binsize=(1-0.2)/lensing_bins


#minzs=np.linspace(0.2,1-redshift_binsize,lensing_bins)


#maxzs=np.linspace(0.2+redshift_binsize,1,lensing_bins)

#lens_boundaries=np.linspace(0.2,1,lensing_bins+1)

'''
CMB lensing efficiency kernel:
'''
def WCMB(chi):
    chi_CMB=cosmo_functions.comoving_distance(1100)
    chi_S=chi_CMB

    #want to return CMB lensing efficiency kernel; see eg eq 4 of 1607.01761
    return 3/2*(H0/c)**2*omegam*chi/cosmo_functions.a(chi)*(1-chi/chi_S)

'''
lensing efficiency kernel (general):
'''
def Wk(chi,chi_S):
    return 3/2*(H0/c)**2*omegam*chi/cosmo_functions.a(chi)*(1-chi/chi_S)
'''
Galaxy lensing efficiency kernel:
'''
zmax=4
Zs=np.linspace(0.01,zmax,2000)
alpha=1.27
beta=1.02
z0=0.5
normalisation_factor_sources=integrate.simps(Zs**alpha*np.exp(-(Zs/z0)**beta),Zs)
    

def dnsource_dz(z): 
    #see section IIC of 1607.01761
    alpha=1.27
    beta=1.02
    z0=0.5
    nsource_total=26
    return nsource_total*z**alpha*np.exp(-(z/z0)**beta)/normalisation_factor_sources 

def number_of_sources_between_zs(z1,z2):
    if(z1>z2):
        zs=np.linspace(z2,z1,100)
    else:
        zs=np.linspace(z1,z2,100)
    return integrate.simps(dnsource_dz(zs),zs)



def equal_numberdensities_26(z2,z1):
    
    
    return number_of_sources_between_zs(min(z1,z2),max(z1,z2))-2.6

def make_redshift_boundaries(z0,zmax):
    boundaries=[z0]
    for i in range(0,10):
        boundaries.append(optimize.brentq(equal_numberdensities_26,boundaries[i]+0.01,zmax,boundaries[i]))
    return boundaries
boundaries_sources=make_redshift_boundaries(0.01,4)
#print("redshift bins for clustering are",boundaries_sources)

'''
Galaxy lensing efficiency kernel
'''
def Wkgal_lensing(chi,i): # i is the ith redshift bin, starting at 0!!!
    
    
    
    zmin=boundaries_sources[i]
    zmax=boundaries_sources[i+1]
    
    z=cosmo_functions.redshift(chi)
    
    if(z>zmax):
        return 0
    
    if(z<zmin):
        zsources=np.linspace(zmin,zmax,10)
    else:
        zsources=np.linspace(z,zmax,10)
    integrand=[dnsource_dz(z)*Wk(chi,cosmo_functions.comoving_distance(z)) for z in zsources]
  #  print("integran dis",integrand)
   # plt.plot(zsources,integrand,'o')
   # plt.show()
   # print( 1/number_of_sources_between_zs(zmin,zmax))
   # print("integral should be",integrate.simps(integrand,zsources))
   # print ("answer should be",1/number_of_sources_between_zs(zmin,zmax),"times",integrate.simps(integrand,zsources),"equals",1/number_of_sources_between_zs(zmin,zmax) *integrate.simps(integrand,zsources))
    return 1/number_of_sources_between_zs(zmin,zmax) *integrate.simps(integrand,zsources) #number of sources should be 2.6 by definition no?
   
def Wkgal_lensing_workingforarrays(chis,i): # i is the ith redshift bin, starting at 0!!!
    
    #this might be sort of hard.
    
    answer=np.zeros(len(chis))
    
    integrand=np.zeros((len(chis),10))
    zmin=boundaries_sources[i]
    zmax=boundaries_sources[i+1]
    
    zs=cosmo_functions.redshift(chis)
    
    answer[zs>zmax]=0
    
    density=1/number_of_sources_between_zs(zmin,zmax)
    
    
    zsources=np.zeros((len(zs),10))
    
    for i in range(0,len(zs)):
        if zs[i]<zmin:
            zsources[i,:]=np.linspace(zmin,zmax,10)
        else:
            zsources[i,:]=np.linspace(zs[i],zmax,10)
            
  

    integrand[zs<zmax,:]=dnsource_dz(zsources[zs<zmax,:])*Wk(chis[zs<zmax,np.newaxis],cosmo_functions.comoving_distance(zsources[zs<zmax,:].flatten()).reshape((len(chis[zs<zmax]),10)))
   # print("integrand is",integrand)
   # print("one over density should be",1/density*integrate.simps(integrand[0],zsources[0]))
    answer[zs<zmax]=density*integrate.simps(integrand[zs<zmax,:],zsources[zs<zmax,:])
 
              
  #  plt.plot(zsources[0],integrand[0],'o')
  #  plt.show()
    return answer#number of sources should be 2.6 by definition no?
   
        


'''
Galaxy density efficiency kernel
'''
Zs=np.linspace(0.2,1,100)
normalisation_factor_lenses=integrate.simps(cosmo_functions.comoving_distance(Zs)**2/ cosmo_functions.results.hubble_parameter(Zs),Zs)

def dnilensdz(z,spec):
    
    if spec=="magic":
        nlens_total=0.25
    
    # is my normalisation factor okay?
        return nlens_total*cosmo_functions.comoving_distance(z)**2/ cosmo_functions.results.hubble_parameter(z)/normalisation_factor_lenses
    elif spec=="LSST":
         ngal=40
         z0=0.3
         return ngal*1/(2*z0)*(z/z0)**2*np.exp(-z/z0)




def N_fields(clustering_bins):
    #where should i put this?
    source_bins=10
   # print(clustering_bins+source_bins)
    return    clustering_bins+source_bins

def bin_boundaries(clustering_bins):
    return np.linspace(0.2,1,clustering_bins+1)



def Wgali_clustering2(chis,i,spec,nbins):
    
    
    '''
    boundaries_lenses=config.lens_boundaries

    #i is the i-th redshift bin, starting at 1, such that i=1 corresponds to (0.2,0.4)
    if(z<boundaries_lenses[i-1]):
        return 0
    if (z>boundaries_lenses[i]):
        
        
        return 0
    '''
    redshift_bin=i-1
 #   #print("in wgali redshift bin is",i-1)
 #   print("spec is",spec)
#    #print("number of bins is",nbins)
#    
    z_bin_boundaries=bin_boundaries(nbins)
    zmin=z_bin_boundaries[redshift_bin]
  #  print(z_bin_boundaries)
   # print(redshift_bin,zmin)
    zmax=z_bin_boundaries[redshift_bin+1]
    
    answer=np.zeros(len(chis))
    
    
    zs=cosmo_functions.redshift(chis)

    answer= dnilensdz(zs,spec)*cosmo_functions.dzdchi(zs) #note WITHOUT bias and WITHOUT normalisation factor of 1/density
      
    answer[zs<zmin]=0
    answer[zs>zmax]=0
   # print(zs,zmin,zmax)
    return answer

def Wgali_clustering(chis,i,spec,nbins):
    
    
    '''
    boundaries_lenses=config.lens_boundaries

    #i is the i-th redshift bin, starting at 1, such that i=1 corresponds to (0.2,0.4)
    if(z<boundaries_lenses[i-1]):
        return 0
    if (z>boundaries_lenses[i]):
        
        
        return 0
    '''
    redshift_bin=i-1
 #   #print("in wgali redshift bin is",i-1)
 #   print("spec is",spec)
#    #print("number of bins is",nbins)
#    
    z_bin_boundaries=bin_boundaries(nbins)
    zmin=z_bin_boundaries[redshift_bin]
  #  print(z_bin_boundaries)
   # print(redshift_bin,zmin)
    
    zmax=z_bin_boundaries[redshift_bin+1]
    
    answer=np.zeros(len(chis))
    
    zs=cosmo_functions.redshift(chis)
  #  print(zs)
   # print(spec)
    answer= dnilensdz(zs,spec)*cosmo_functions.dzdchi(zs) #note WITHOUT bias and WITHOUT normalisation factor of 1/density
      
    answer[zs<zmin]=0
    answer[zs>zmax]=0
   # print(zs,zmin,zmax)
   
    return answer
def W(spec,chis,nbins,experiment=None):
    
    
    
    if spec=="CMB lensing":
        return np.array([WCMB(chi)for chi in chis])
    
    if spec[0:5]=="shear":
        i=int(spec[6:])
        return Wkgal_lensing_workingforarrays(chis,i)
    
    if spec[0:7]=="density":
        i=int(spec[8:])
        #return np.array([Wgali_clustering(chi,i,experiment)for chi in chis])
        return Wgali_clustering(chis,i,experiment,nbins)
    else:
        print("spec problem")
        
        
