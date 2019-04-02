#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 03:16:32 2019

@author: fionamccarthy
"""

import numpy as np

import matplotlib.pyplot as plt


from scipy.interpolate import interp1d
from scipy import integrate
from scipy import optimize

import sys
import cosmo_functions

import lensing_kernels

import time

H0=cosmo_functions.H0
c=cosmo_functions.c
omegam=cosmo_functions.omegam

#lensing_bins=config.lensing_bins



Pnonlin=cosmo_functions.Pnonlin


'''

I want to evaluate the Kernels in each CLUSTERING bin on some grid with grid points (zpoints).

I think I would like the number of redshift bins to be a parameter I can PUT IN to the code.

'''

source_bins=10





def bin_boundaries(clustering_bins):
    return np.linspace(0.2,1,clustering_bins+1)

#upperlimits=[2*np.pi*cosmo_functions.comoving_distance(minz)/10*H0/100 for minz in minzs]
def N_fields(clustering_bins):
    return    clustering_bins+source_bins



def Cls(spec,ls,resolution,number_of_clustering_bins,experiment=None):
    
    if spec[0]=="shear":
        shears=["shear_"+str(i)for i in range(0,source_bins)]
    
        shear_powerspectra=np.zeros((source_bins,source_bins,len(ls)))
    
        Ws=np.zeros((source_bins,resolution)) #want to evaluate the redshift kernel FOR EACH SOURCE BIN AT EACH Z SAMPLE
                                              #set up the Ws by evaluating at each z      
        zs=np.linspace(0.01,4,resolution)
        chis=cosmo_functions.comoving_distance(zs)
        for i in range(0,source_bins):
                 Ws[i,:]=lensing_kernels.W(shears[i],chis)   
                 
                 
        for i in range(0,source_bins):
            Ws1=Ws[i]
            for j in range(0,source_bins):
                Ws2=Ws[j]
                if j>=i:
                    shear_powerspectra[i,j]=shear_powerspectra[j,i]=Cls_limber(ls,Ws1,Ws2,Pnonlin,chis,zs)
                        
    
        return shear_powerspectra
    elif spec[0]=="clustering":
        
    
        N_field=N_fields(number_of_clustering_bins)
        bini_powerspectra=np.zeros((N_field,len(ls)))

        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        redshift_bin=spec[1]
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]

        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))
        
        

        zs=np.linspace(zmin,zmax,resolution)
        chis=cosmo_functions.comoving_distance(zs)
        W_clustering=np.zeros(len(chis))
        
        normalisation=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)
        
        W_clustering=galaxy_bias*lensing_kernels.W("density_"+str(redshift_bin+1),chis,experiment)/normalisation
        
        Ws=np.zeros((source_bins,resolution))
        
        
        for i in range(0,source_bins):
                if(zmin<lensing_kernels.boundaries_sources[i+1]) :  #want the lenses to be BEHIND the clustering galaxies
                    Ws[i,:]=lensing_kernels.W("shear_"+str(i),chis) 
                    
                bini_powerspectra[number_of_clustering_bins+i,:]=Cls_limber(ls,W_clustering,Ws[i],Pnonlin,chis,zs)
        
        bini_powerspectra[redshift_bin,:]=Cls_limber(ls,W_clustering,W_clustering,Pnonlin,chis,zs)
        
        return bini_powerspectra
    else:
        print("spec problem; spec is",spec)
        return 

        
        
        

def Signal_Covariance_Matrix_sparse(ls,zresolution_clusteringbins,zresolution_shearbins,number_of_clustering_bins,experiment):
    
    N_field=N_fields(number_of_clustering_bins)

    Cl_sparse=np.zeros((N_field,N_field,len(ls)))
    
    print("getting shear power spectra")
    
    start=time.time()
    Cl_sparse[number_of_clustering_bins:,number_of_clustering_bins:,:]=Cls(["shear"],ls,zresolution_shearbins,number_of_clustering_bins)

    end=time.time()
    
    print("got shear power spectra in ",end-start," seconds")
    
    print("getting clustering power spectra")
    start1=time.time()
    for redshift_bin in range(0,number_of_clustering_bins):
            start=time.time()


            Cl_sparse[redshift_bin,:,:]=Cl_sparse[:,redshift_bin,:]=Cls(["clustering",redshift_bin],ls,zresolution_clusteringbins,number_of_clustering_bins,experiment)
            end=time.time()
    end1=time.time()
    print("got clustering power spectra in ", end1-start1,"seconds")

    

    return Cl_sparse

    

def Cls_limber(ls,Ws1,Ws2,matter_pk,chis,zs):#specs tells you which power spectrum you want
   
    #ls is the l-values you will compute at
    #Ws1 and Ws2 are the kernels you are integrating
   
    #zs=redshift(chis)
    
    Cls=[]
    Ws1=np.array(Ws1)  #maybe pass in arrays?? compute these as arrays?
    Ws2=np.array(Ws2)
    zs=np.array(zs)
    chis=np.array(chis)  
    int1= Ws1 * Ws2 /chis**2 
    for l in ls:
        #integrand=[Ws1[k]*Ws2[k]*matter_pk(zs[k],(l+1/2)/chis[k])/chis[k]**2 for k in range(0,len(chis))]
        
        int2=np.array([matter_pk(zs[k],(l+1/2)/chis[k]) for k in range(0,len(chis))]) #would be cool if i could get this to work for arrays?
        #int2=matter_pk(zs,(l+1/2)/chis).diagonal()
        integrand=int1*int2
        Cls.append(integrate.simps(integrand,chis))
        
    return np.array(Cls)




def clustering_noise(redshift_bin,zresolution,number_of_clustering_bins,experiment):
    z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
    
    zmin=z_bin_boundaries[redshift_bin]
    
    zmax=z_bin_boundaries[redshift_bin+1]

        
    
    zs=np.linspace(zmin,zmax,zresolution)
    
    density_insr=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)*(np.pi/180)**-2*60**2 #arcmin^-2 -> sr^-1
    noiseval_lenses=1/density_insr
    return noiseval_lenses


def shear_noise(redshift_bin):
     density_insr=lensing_kernels.number_of_sources_between_zs(lensing_kernels.boundaries_sources[redshift_bin],lensing_kernels.boundaries_sources[redshift_bin+1])*(np.pi/180)**-2*60**2 
     noiseval_sources= 0.26 **2/density_insr #see pg 4 of 1607.01761. should be squared?
     return noiseval_sources
    
def clustering_Noise_Covariance_Matrix_sparse(ls,zresolution_clusteringbins,number_of_clustering_bins,experiment):
    N_field=N_fields(number_of_clustering_bins)


    Nl_sparse=np.zeros((N_field,N_field,len(ls)))
    for i in range(0,number_of_clustering_bins):
        
        
        Nl_sparse[i,i,:]=clustering_noise(i,zresolution_clusteringbins,number_of_clustering_bins,experiment)
    return Nl_sparse

def lensing_Noise_Covariance_Matrix_sparse(ls,zresolution_clusteringbins,number_of_clustering_bins):
    N_field=N_fields(number_of_clustering_bins)


    Nl_sparse=np.zeros((N_field,N_field,len(ls)))
    for i in range(number_of_clustering_bins,N_field):
        
        
        Nl_sparse[i,i,:]=shear_noise(i-number_of_clustering_bins)
    return Nl_sparse
    
def Noise_Covariance_Matrix_sparse(ls,zresolution_clusteringbins,number_of_clustering_bins,experiment):

    N_field=N_fields(number_of_clustering_bins)


    Nl_sparse=np.zeros((N_field,N_field,len(ls)))

    for i in range(number_of_clustering_bins,N_field):
        
        
        Nl_sparse[i,i,:]=shear_noise(i-number_of_clustering_bins)
    
    for i in range(0,number_of_clustering_bins):
        
        
        Nl_sparse[i,i,:]=clustering_noise(i,zresolution_clusteringbins,number_of_clustering_bins,experiment)
    return Nl_sparse

def dCldb(covariance_matrix,number_of_clustering_bins): #param is the parameter you are differentiating wrt
    
    
    dCldb=np.zeros((number_of_clustering_bins,covariance_matrix.shape[0],
                         covariance_matrix.shape[1],covariance_matrix.shape[2]))
    
    for redshift_bin in range(0,number_of_clustering_bins):
        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]

        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))    
        dCldb[redshift_bin,redshift_bin,redshift_bin,:]=2*covariance_matrix[redshift_bin,redshift_bin,:]/galaxy_bias
        dCldb[redshift_bin,number_of_clustering_bins:,redshift_bin,:]= dCldb[redshift_bin,redshift_bin,number_of_clustering_bins:,:]=covariance_matrix[redshift_bin,number_of_clustering_bins:,:]
        
    return dCldb

def Cls_fnlderivatives(redshift_bin,ls,zresolution_clusteringbins,number_of_clustering_bins,experiment):
        start=time.time()
        N_field=N_fields(number_of_clustering_bins)
        bini_powerspectrafnlderivs=np.zeros((N_field,len(ls)))

        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]

        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))
        
        

        zs=np.linspace(zmin,zmax,zresolution_clusteringbins)
        chis=cosmo_functions.comoving_distance(zs)
        W_clustering=np.zeros(len(chis))
        
        normalisation=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)
        
        W_clustering=galaxy_bias*lensing_kernels.W("density_"+str(redshift_bin+1),chis,experiment)/normalisation
        Ws=np.zeros((source_bins,zresolution_clusteringbins))
        Cl_fnlderivs=np.zeros(len(ls))
        timef=time.time()
        fnl_factors=np.zeros((len(chis),len(ls)))
        for z_index in range(0,len(chis)):
            for l_index in range(0,len(ls)):
                k=(ls[l_index]+1/2)/chis[z_index]
                fnl_factors[z_index,l_index]=cosmo_functions.fnl_bias_factor(k,chis[z_index])
        timef2=time.time()
        
        for ell_index,l in enumerate(ls):
                Ks=(l+1/2)/chis
                
                
                integrand=[2*W_clustering[k]*W_clustering[k]*
                           Pnonlin(zs[k],(l+1/2)/chis[k])/chis[k]**2
                           *1/galaxy_bias*
                           1/Ks[k]**2
                           *(galaxy_bias-1)
                           *fnl_factors[k,ell_index] for k in range(0,len(chis))]
                Cl_fnlderivs[ell_index]=(integrate.simps(integrand,chis))
            #    plt.plot(chis,integrand,'o')
  #              plt.loglog(chis,integrand,'o')
#            #    plt.show()
 #               plt.show()
        bini_powerspectrafnlderivs[redshift_bin,:]=Cl_fnlderivs
        end=time.time()
        for i in range(0,source_bins):
            if(zmin<lensing_kernels.boundaries_sources[i+1]) :  #want the lenses to be BEHIND the clustering galaxies
                Ws[i,:]=lensing_kernels.W("shear_"+str(i),chis) 
            
            Cl_fnlderivs=np.zeros(len(ls))
            
            
            
            for ell_index,l in enumerate(ls):
                Ks=(l+1/2)/chis
                integrand=[Ws[i,k]*W_clustering[k]*
                           Pnonlin(zs[k],(l+1/2)/chis[k])/chis[k]**2
                           *1/galaxy_bias*
                           1/Ks[k]**2
                           *(galaxy_bias-1)
                           *fnl_factors[k,ell_index] for k in range(0,len(chis))]
        
                Cl_fnlderivs[ell_index]=(integrate.simps(integrand,chis))
            
            bini_powerspectrafnlderivs[number_of_clustering_bins+i,:]=Cl_fnlderivs
        end2=time.time()
        
        
        return bini_powerspectrafnlderivs
    
    

def dCldfnl(number_of_clustering_bins,zresolution_clusteringbins,ls,experiment):
    
    N_field=N_fields(number_of_clustering_bins)

    dCldfnl=np.zeros((N_field,N_field,len(ls)))
    
   
    
    print("getting clustering power spectra derivatives")
    start1=time.time()
    for redshift_bin in range(0,number_of_clustering_bins):
            start=time.time()


            dCldfnl[redshift_bin,:,:]=dCldfnl[:,redshift_bin,:]=Cls_fnlderivatives(redshift_bin,ls,zresolution_clusteringbins,number_of_clustering_bins,experiment)
            end=time.time()
    end1=time.time()
    print("got clustering power spectra derivatives in ", end1-start1,"seconds")

    
    return dCldfnl
    

def dCl_sparse(ls,zresolution_clusteringbins,zresolution_shearbins,number_of_clustering_bins,covariance_matrix,experiment):
    
    
    dCl_sparse=np.zeros((1+number_of_clustering_bins,covariance_matrix.shape[0],covariance_matrix.shape[1],covariance_matrix.shape[2]))
    
    
    
    dCl_sparse[1:,:,:,:]=dCldb(covariance_matrix,number_of_clustering_bins)
    
    dCl_sparse[0,:,:,:]=dCldfnl(number_of_clustering_bins,zresolution_clusteringbins,ls,experiment)
    
    return dCl_sparse

def size_of_covmatrix(ls,ell_index,clustering_bin_number):
    
    
            l=ls[ell_index]
            
            boundaries=bin_boundaries(clustering_bin_number)
            upperlimits=[2*np.pi*cosmo_functions.comoving_distance(minz)/10*H0/100 for minz in boundaries[:-1]]
            
            index=0
            
            while(index<clustering_bin_number):
            
                if l<upperlimits[index]:
                    return clustering_bin_number+10-index
                else:
                    index=index+1
                    
            return 10

        

def fisher_matrix(ls,signal_covariance,lens_noise,clustering_noise,dcdalpha,number_of_clustering_bins): #if discrete is false I will integrate over l instead of sum
    
        Fij_ells=np.zeros((1+number_of_clustering_bins,1+number_of_clustering_bins,len(ls)))
    
        for ell_index in range(0,len(ls)):
            sys.stdout.write(str(ell_index/ls[-1]*100)+"%\r")            
            N=size_of_covmatrix(ls,ell_index,number_of_clustering_bins)   #imposes maximum l
            '''
            Cl=covariance[-N: , -N: , ell_index]  # deleting the first row and column if N=13,
                                              # the second row and column if N=12, ec
            Nl=clustering_noise[-N:, -N:, ell_index]+lens_noise[-N: , -N:, ell_index]
            '''
                                              # print(noise_lensing[-N: , -N:, ell_index])
            
            
            covariance_mat=signal_covariance[-N:,-N:,ell_index]+lens_noise[-N:,-N:,ell_index]+clustering_noise[-N:,-N:,ell_index]
            inverse_covm=np.linalg.inv(covariance_mat)


            for i in range(0,Fij_ells.shape[0]):
        
                for j in range(0,Fij_ells.shape[0]):
        
            
                    first=np.matmul(dcdalpha[i,-N:,-N:,ell_index],inverse_covm)
                    second=np.matmul(dcdalpha[j,-N:,-N:,ell_index],inverse_covm)
                
                    Fij_ells[i,j,ell_index]=np.trace(np.matmul(first,second))
                
        Fisher=np.zeros((1+number_of_clustering_bins,1+number_of_clustering_bins))
                
        for i in range(0,1+number_of_clustering_bins):
        
            for j in range(0,1+number_of_clustering_bins):
         
               
                    Fisher[i,j]=np.sum([(2*ls[k]+1)*Fij_ells[i,j,k] for k in range(0,len(ls))])
               
        return(18000/41253 *Fisher/(2))  
        
def interpolate_covariancematrix(ell_sparse,all_ell,cm_sparse):
        
        all_cm=np.zeros((cm_sparse.shape[0],cm_sparse.shape[1],len(all_ell)))
        for i in range(0,cm_sparse.shape[0]):
            for j in range(0,cm_sparse.shape[1]):
                interp=interp1d(ell_sparse,cm_sparse[i,j,:])
                all_cm[i,j,:]=interp(all_ell)
        return all_cm
    
def interpolate_dcdalpha(ell_sparse,all_ell,dcdalpha_sparse,all_cm,number_of_clustering_bins):
        
    dcdalpha_all=np.zeros((dcdalpha_sparse.shape[0],dcdalpha_sparse.shape[1],dcdalpha_sparse.shape[2],len(all_ell)))
    
    dcdalpha_all[1:,:,:,:]=dCldb(all_cm,number_of_clustering_bins)

    dcdfnl_all=np.zeros((dcdalpha_sparse.shape[1],dcdalpha_sparse.shape[2],len(all_ell)))
    

    for i in range(0,dcdfnl_all.shape[0]):
        for j in range(0,dcdfnl_all.shape[1]):
                interp=interp1d(ell_sparse,dcdalpha_sparse[0,i,j,:])
                dcdfnl_all[i,j,:]=interp(all_ell)
                
    dcdalpha_all[0,:,:,:]=dcdfnl_all
    return dcdalpha_all
        
        