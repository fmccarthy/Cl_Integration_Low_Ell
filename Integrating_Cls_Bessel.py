#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:27:37 2019

@author: fionamccarthy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:15:25 2019

@author: fionamccarthy
"""

import numpy as np
#import matplotlib.pyplot as plt

from scipy.interpolate import interp1d,interp2d
from scipy import integrate, optimize, special

import time

import cosmo_functions
import lensing_kernels

test=False

#cosmological parameters
H0=cosmo_functions.H0
c=cosmo_functions.c
omegam=cosmo_functions.omegam



def coeffs(function,Nmax,kmin,kmax,bias,test=False):
    #freturns ourier transform coefficients and frequencies
    #parameters: function = function that will be transformed 
    #            Nmax = number of Fourier frequencies required
    #            kmin = minimum k 
    #            kmax = maximum k
    #            bias = bias to transform function
    
    #note if this needs to be optimized can probably save a factor of 2 by only computing
    #half the coeffs  / freqs and taking complex conjugate
    
    ##### see 1705.05022 for further details!
    
    kappamin,kappamax=np.log(kmin),np.log(kmax)
    
    ints=np.linspace(-int(Nmax/2),int(Nmax/2),Nmax+1-Nmax%2)

    period=kappamax-kappamin
    kappas=kappamin+np.linspace(0,period,Nmax+1-Nmax%2)
    samples=function(np.exp(kappas))*(np.exp(kappas)/np.sqrt(kmin*kmax))**bias #sampling the function at each k
      
    cns=np.zeros(len(ints),dtype=np.complex_)
    fns=np.zeros(len(ints),dtype=np.complex_)
    
    fns=2*np.pi*(0+1j)*ints/period-bias
    cns=1/Nmax*np.sum(samples[:,np.newaxis]*np.exp(-(2*np.pi*(0+1j)*ints*ints[:,np.newaxis])/Nmax),axis=0) *1/(np.sqrt(kmin*kmax))**fns
   
    if Nmax%2==0:
        cns[0]=cns[0]/2
        cns[-1]=cns[-1]/2
    return cns,fns

def logf_transform(function,x,Nmax,xmin,xmax,bias,test=False):
    #makes the log fourier series
    #parmeters: function = function that will be transformed
    #           x = value evaluating the function f(x)
    #           Nmax = number of Fourier frequencies reauried
    #           xmin = minimum x value
    #           xmax = maximum x value (function is periodic in log x with this period)
    #           bias = bias to transform with
    
    cns,fns=coeffs(function,Nmax,xmin,xmax,bias,test)
    return np.sum([cns[i]*((x/1))**(fns[i]) for i in range(0,len(cns))])



def hypf(a,b,c,z):
    #hypergeometric function defined through its power series representation
    
    #note this is not a good idea to use for abs(z) near 1, it will take a long time to converge.
    if np.max(z)>0.7:
        print("check you are using the correct hypergeometric function! hypfnearone should be used for z near 1. will take a while to compute for z=",z[z>0.8])
    
    term=1
    epsilon=1
    answer=0
    n=0
    while(np.max(epsilon)>1e-10): #computes on arrays until ALL elements have converged here
        previousanswer=answer
        answer=answer+term #starts at 1
        term=term*(a+n)*(b+n)/((c+n)*(n+1))*z 
        epsilon=np.abs((answer-previousanswer)/answer) #checks convergence
        n=n+1
    return answer
    

def hypfnearone(ell,nu,t):
    #hypergemetric function evaluated near one as needed to compute t near 1. see appendxix B of 1705.05022
    
    ##equation B5 of 1705.05022 as equation b6 did not work for me
    
     ttilde=-(1-t**2)**2/(4*t**2)
     xx=special.gamma(ell+3/2)*t**(-ell-nu/2)*(special.gamma(nu-2)*t**(nu-2)*(1-t**2)**(2-nu)/(special.gamma((nu-1)/2)*special.gamma(ell+nu/2))*hypf((2*ell-nu+4)/4,(-2*ell-nu+2)/4,2-nu/2,ttilde)+special.gamma(2-nu)/(special.gamma(3/2-nu/2)*special.gamma(ell-nu/2+2))*hypf((2*ell+nu)/4,(-2*ell+nu-2)/4,nu/2,ttilde))
     return xx
     

def i_ell_floats(ell,nu,t):
    
    #do i still need this?
    
    tstar=0.7
    if t<tstar:
        return 2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*t**ell* hypf((nu-1)/2,ell+nu/2,ell+3/2,t**2)
    else:
       
        return 2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*t**ell*hypfnearone(ell,nu,t)



def i_ell_tarray(ell,nus,ts):
    
    
    answer=np.zeros((len(ell),len(ts),len(nus)),dtype=np.complex_)
    hypfs=np.zeros((len(ell),len(ts),len(nus)),dtype=np.complex_)
    
    tstar=0.7    #above tstar we compute i_ell a different way due to slow convergence for t near 1.

    
    hypfs[:,ts<tstar,:]=hypf((nus-1)/2,ell[:,np.newaxis,np.newaxis]+nus/2,ell[:,np.newaxis,np.newaxis]+3/2,ts[ts<tstar,np.newaxis]**2)
    
    hypfs[:,ts>tstar,:]=hypfnearone(ell[:,np.newaxis,np.newaxis],nus,ts[ts>tstar,np.newaxis])
    
    answer[:,ts<tstar,:]= 2**(nus-1)*np.pi**2*special.gamma(ell[:,np.newaxis,np.newaxis]+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2)[:,np.newaxis,np.newaxis])* hypfs[:,ts<tstar,:]*ts[ts<tstar,np.newaxis]**ell[:,np.newaxis,np.newaxis]
    
    answer[:,ts>tstar,:]=2**(nus-1)*np.pi**2*special.gamma(ell[:,np.newaxis,np.newaxis]+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2)[:,np.newaxis,np.newaxis])*hypfs[:,ts>tstar,:]*ts[ts>tstar,np.newaxis]**ell[:,np.newaxis,np.newaxis]
    
    return answer



def i_ell(ell,nus,tresolution):
    
    #this is pretty messy but it works..
    
    #I need to compute i_ell at DIFFERENT ts for each (ell, nu) because there is a different
    #t_min for each and I am computing always at (tresolution) ts
    
    mints=interpolated_minT(ell,nus) #so first of all we find the minimum t at which to cmopute
    ts=np.zeros((len(nus),tresolution))
    
    tstar=0.7
    for nu_index, nu in enumerate(nus):
        #is there a way to do this using array manipulation? eg making an array of linspaces? come back to this
        tt=np.linspace(mints[nu_index],1-1e-5,tresolution) #now we make an array of 50 ts starting from t_min. 
        ts[nu_index]=tt
    answer=np.zeros((len(nus),tresolution),dtype=np.complex_) #to store the answer
    
    hypfs_lessthan=np.zeros((len(nus),tresolution),dtype=np.complex_)  #where we compute the hypergeometric functions near zero
    hypfs_greaterthan=np.zeros((len(nus),tresolution),dtype=np.complex_)#ditto except near one
    
    answer_greaterthan=np.zeros((len(nus),tresolution),dtype=np.complex_)
    answer_lessthan=np.zeros((len(nus),tresolution),dtype=np.complex_)
    
    ts_lessthan=np.zeros(ts.shape)      # i don't want to waste time trying to compute hypfs near one so i will store the ts<tstar in this array where i will later change the high ts to something small, and discard the relevant entries.
    ts_greaterthan=np.zeros(ts.shape)   # ditto
    
    ts_lessthan[:,:]=ts.copy()
    ts_greaterthan[:,:]=ts.copy()
    
    # replacing all the t values near 1 with some arbitrary number that won't take ages to converge:
    ts_lessthan[ts>tstar]=0.2              
    ts_greaterthan[ts<tstar]=0.9
    #compute relevant hypergeometric functions:
    hypfs_lessthan=hypf((nus[:,np.newaxis]-1)/2,ell+nus[:,np.newaxis]/2,ell+3/2,ts_lessthan**2)
    hypfs_greaterthan=hypfnearone(ell,nus[:,np.newaxis],ts_greaterthan)
     #putting them into the answer (i probably don't need to do the first computation twice, it can probably go at the end...)
    answer_lessthan=((2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2))))[:,np.newaxis]*ts**ell* hypfs_lessthan
    answer_greaterthan=(2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2)))[:,np.newaxis]*ts**ell*hypfs_greaterthan
    
    #discarding the irellevant answers:
    answer_lessthan[ts>tstar]=0
    answer_greaterthan[ts<tstar]=0
    
    #combining:
    answer=answer_lessthan+answer_greaterthan

    return answer


def N_fields(clustering_bins):
    #where should i put this?
    return    clustering_bins+source_bins

def bin_boundaries(clustering_bins):
    return np.linspace(0.2,1,clustering_bins+1)

def Wg(chis, FIELD,number_of_clustering_bins,experiment="magic",test=False):
    #where we compute the window function 
    #parameters: chis = comoving distance array to evaluate at
    
    #           FIELD = A LIST ["TYPE",BIN_NUMBER]
    #                   WHERE "TYPE" IS EITHER "density" or "shear" 
    #                   and BIN_NUMBER is the relevant bin number starting indexing at 0. 
    #           experiment = "magic" or "LSST" depending on how dense the galaxies are    
    
    #remember the power spectrum we transform is P(z=0,k) so we must multiply any Wg by the growth factor normalized to 1 at z=0!!
    
    
    if test:
        sigma=300
        chiav=3000
        Wg=np.zeros(len(chis))
        Wg[chis<8000]= 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(chis[chis<8000]-chiav)**2/(2*sigma**2))*cosmo_functions.D_growth_chih_normz0(chis[chis<8000])
        return Wg
  
    
    redshift_bin=FIELD[1]

    if FIELD[0]=="density":
        

        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]
        
        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))
        
        zs=np.linspace(zmin,zmax,100)#is this enough resolution
        normalisation=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)
        
        W=galaxy_bias*lensing_kernels.W("density_"+str(redshift_bin+1),chis,number_of_clustering_bins,experiment)/normalisation
        
        
    if FIELD[0]=="shear":
        #check this. maybe i should multiply by 1/chi in the integral before finding the bin-averaged kernel?
        W=lensing_kernels.W("shear_"+str(redshift_bin),chis,number_of_clustering_bins)*1/(chis)**2 #multiply by 1/chi because this has been incorporated into the limber approximation in lensing-kernels.
        
    return W*cosmo_functions.D_growth_chi_normz0(chis) #include growth factor in definition of Wg.
 
def firstderiv_Wg(chi,FIELD,number_of_clustering_bins,test=False):
    #compute the first derivative 
    
    deltachi=chi*0.000001
    return (Wg(chi+deltachi,FIELD,number_of_clustering_bins,"magic",test)-Wg(chi-deltachi,FIELD,number_of_clustering_bins,"magic",test))/(2*deltachi)

def secondderiv_wg(chi,FIELD,number_of_clustering_bins,test=False):
    
    deltachi=chi*0.000001
    return (Wg(chi-deltachi,FIELD,number_of_clustering_bins,"magic",test)+Wg(chi+deltachi,FIELD,number_of_clustering_bins,"magic",test)-2*Wg(chi,FIELD,number_of_clustering_bins,"magic",test))/(deltachi**2)

    
def dl_wg_ell_independent(chi,FIELD,number_of_clustering_bins,test=False):
    
     #there is a derivative operator defined in equation 2.24 of 1705.05022. This computes the ell-independent part of it.
     return -secondderiv_wg(chi,FIELD,number_of_clustering_bins,test)+2/chi*firstderiv_Wg(chi,FIELD,number_of_clustering_bins,test)
 
    
#now i interpolate the dl_wg_ell_independent because it would take a lONG time to compute the derivative numerically each time!!
chis=np.linspace(1,13000,1000) #is this enough chi-sampling? I should change this.

number_of_clustering_bins=4 #change this to a parameter???


if not test:
    W_interp_densities=[interp1d(chis,Wg(chis,["density",i,"magic"],number_of_clustering_bins,"magic",test))for i in range(0,number_of_clustering_bins)]
    W_interp_shears=[interp1d(chis,Wg(chis,["shear",i,"magic"],number_of_clustering_bins,"magic",test))for i in range(0,10)]

print("done first interp")
interp_dlwg_densities=[interp1d(chis,dl_wg_ell_independent(chis,["density",i,"magic"],number_of_clustering_bins,test))for i in range(0,number_of_clustering_bins)]
#i only need this for the density field 

print("done derivative interpolations")

def interpolated_Wg(chis,FIELD):
    #returns the appropriate interpolated window function. maybe i should put this in lensing_kernels??
    
    if FIELD[0]=="density":     
        return W_interp_densities[FIELD[1]](chis)
    else:   
        answer=np.zeros(chis.shape)
        chis[chis<1]=1
        answer[chis<13000 ]= W_interp_shears[FIELD[1]](chis[chis<13000])
      #  answer[chis]
        return answer

def interp_dlwg_first(chis,FIELD):
    
      #returns the apprppriate interpolated first (ell-independent) part of the dl_wg operator acting on the appropriate field. 

    if FIELD[0]=="density":
          answer=np.zeros(chis.shape)
          answer[chis>1]=interp_dlwg_densities[FIELD[1]](chis[chis>1])
          
          return answer
    else:
        print("should not need the Dl of a shear field")
        return
 
def Dl_wg(chi,ell,FIELD,experiment):
    
    #operator in 2.24 of 1705.05022
    
    #interp_dlwg_first is  -d^2d/chi^2 +2/chi d/dchi. 
    ans=np.zeros(chi.shape)
    chi[chi<1]=1
    
    ans[chi<13000]=interp_dlwg_first(chi[chi<13000],FIELD)+(ell*(ell+1)-2)/chi[chi<13000]**2*interpolated_Wg(chi[chi<13000],FIELD)
    return ans



def intWgalaxy_lowell(ells,nus,ts,chiresolution,SPECTRUM,nbins,experiment,test=False):
    if np.max(ells>11):
        print("using wrong intwgalaxy at ell =",ells)
    #parameters: ells = array of ells. note ELLS MUST BE BELOW 11
    #            nus = array of frequencies
    #            ts = array of ts
    #            SPECTRUM = relevant spectrum you want to compute. 
    #                       should be a list of two FIELDs [FIELD1, FIELD2]
    #                       where FIELD is a list of [type, redshift_bin] and 
    #                       type is either "density" or "shear"
    
    
    answer=np.zeros((len(ells),len(ts),len(nus)),dtype=np.complex_)
    
    FIELD1=SPECTRUM[0]
    FIELD2=SPECTRUM[1]
    redshift_bin=FIELD1[1]

    if FIELD1[0]=="density":
        if FIELD2[0]=="density":
            z_bin_boundaries=bin_boundaries(nbins)
    
            zmin=z_bin_boundaries[redshift_bin]
            zmax=z_bin_boundaries[redshift_bin+1]
            chimin=cosmo_functions.comoving_distance(zmin) 
    
        #the window function is exactly zero outside of this range so this should be okay.
    
            chimax=cosmo_functions.comoving_distance(zmax) 
            '''
            if test:
                chimin = 3000 - 5*300
                chimax=3000+5*300
            '''
        
            chis=np.linspace(chimin,chimax,chiresolution)
            t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array

            for ell_index,ell in enumerate (ells):
            #any point in making this an array operation? I don't think it will speed up that much and there must be some reason i didnt... 
            #I think ell x t x nu x chiresolution is just too big for my computer to handle.
            #regardless there are only about 10 ells here.
    
                first=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2,experiment)) #chi times t
    
                second=Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2,experiment)#chi times t

                secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
                                                                                                 #would like a chi-times-nu-times-t shaped array
    
                firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
    
                total=(Dl_wg(chis,ell,FIELD1,experiment)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
    
            
                total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
                        
                answer[ell_index]=integrate.simps(total_integrand,chis,axis=0)
        elif FIELD2[0]=="shear":
            
            z_bin_boundaries=bin_boundaries(nbins)
    
            zmin=z_bin_boundaries[redshift_bin]
            zmax=z_bin_boundaries[redshift_bin+1]
            chimin=cosmo_functions.comoving_distance(zmin) 
    
        #the window function is exactly zero outside of this range so this should be okay.
    
            chimax=cosmo_functions.comoving_distance(zmax) 
            '''
            if test:
                chimin = 3000 - 5*300
                chimax=3000+5*300
            '''
        
            chis=np.linspace(chimin,chimax,chiresolution)
            t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array

            for ell_index,ell in enumerate (ells):
            #any point in making this an array operation? I don't think it will speed up that much and there must be some reason i didnt... 
            #I think ell x t x nu x chiresolution is just too big for my computer to handle.
            #regardless there are only about 10 ells here.
    
                first=interpolated_Wg(chis[:,np.newaxis]*ts,FIELD2) #chi times t
     
           
    
                second=(interpolated_Wg(chis[:,np.newaxis]/ts,FIELD2)) #chi times 
            
                                                                                      #would like a chi-times-nu-times-t shaped array
          
                secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
           
                firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
            
                total=(Dl_wg(chis,ell,FIELD1,experiment)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
   
                total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
                    
                answer[ell_index]=integrate.simps(total_integrand,chis,axis=0)

            
    elif FIELD1[0]=="shear":
        if FIELD2[0]=="shear":
            chimin=1#what should this be?
            chimax=cosmo_functions.comoving_distance(lensing_kernels.boundaries_sources[redshift_bin+1])
            chis=np.linspace(chimin,chimax,chiresolution)
            t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array
            for ell_index,ell in enumerate (ells):
            #any point in making this an array operation? I don't think it will speed up that much and there must be some reason i didnt... 
            #I think ell x t x nu x chiresolution is just too big for my computer to handle.
            #regardless there are only about 10 ells here.
            
                first=interpolated_Wg(chis[:,np.newaxis]*ts,FIELD2) #chi times t
     
           
    
                second=(interpolated_Wg(chis[:,np.newaxis]/ts,FIELD2)) #chi times 
            
                                                                                      #would like a chi-times-nu-times-t shaped array
          
                secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
           
                firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
            
                total=(interpolated_Wg(chis,FIELD1)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
   
                total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
                    
                answer[ell_index]=integrate.simps(total_integrand,chis,axis=0)
                
        elif FIELD2[0]=="density":
            print("deal with shear/density")
        
    return answer #an ellxt -times nu shaped array.

def intWgalaxy_highell(ell,nus,tresolution,chiresolution,SPECTRUM,nbins,experiment,test=False):
    
    
    #because we have DIFFERENT t-arrays to integrate over at each (ell,nu) it is most convenient to put this in a different
    #function than at low ell.
    
    #parameters: ell = one ell, note ELLS MUST BE ABOVE 10
    #            nus = array of frequencies
    #            tresolution = number of ts to be integrated over
    #            SPECTRUM = relevant spectrum you want to compute. 
    #                       should be a list of two FIELDs [FIELD1, FIELD2]
    #                       where FIELD is a list of [type, redshift_bin] and 
    #                       type is either "density" or "shear"    
    mints=interpolated_minT(ell,nus)
    redshift_bin=SPECTRUM[0][1]
    FIELD1=SPECTRUM[0]
    FIELD2=SPECTRUM[1]
    if FIELD1[0]=="density":
        if FIELD2[0]=="density":

            z_bin_boundaries=bin_boundaries(nbins)
    
            zmin=z_bin_boundaries[redshift_bin]
            zmax=z_bin_boundaries[redshift_bin+1]
            
            chimin=cosmo_functions.comoving_distance(zmin) #change this
            chimax=cosmo_functions.comoving_distance(zmax)#change this
          
            chis=np.linspace(chimin,chimax,chiresolution)
    
            t_independen_multiplicitave_factor=chis**(1-nus[:,np.newaxis])*Dl_wg(chis,ell,FIELD1,experiment)#returns chi-shaped array
    
            answer=np.zeros((len(nus),tresolution),dtype=np.complex_)
            for nu_index,nu in enumerate(nus):
                #it is too much for my computer to handle doing this as an array because nu x tresolution x chiresolution is crazy big
                mint=mints[nu_index]
                ts=np.linspace(mint,1-1e-5,tresolution)
                t_dependent_part=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2,experiment)+ts**(nu-2)*Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2,experiment))#hopefully a chi *t shaped array
                
                total_integrand=t_independen_multiplicitave_factor[nu_index,:,np.newaxis]*t_dependent_part
    
                answer[nu_index]=integrate.simps(total_integrand,chis,axis=0)
          
                #takes around 0.3 seconds
        elif FIELD2[0]=="shear":
            
    
            z_bin_boundaries=bin_boundaries(nbins)
    
            zmin=z_bin_boundaries[redshift_bin]
            zmax=z_bin_boundaries[redshift_bin+1]
            
            chimin=cosmo_functions.comoving_distance(zmin) #change this
            chimax=cosmo_functions.comoving_distance(zmax)#change this
            '''
            if test:
                chimin = 3000 - 5*300
                chimax=3000+5*300
            '''
            chis=np.linspace(chimin,chimax,chiresolution)
    
            t_independen_multiplicitave_factor=chis**(1-nus[:,np.newaxis])*Dl_wg(chis,ell,FIELD1,experiment)#returns chi-shaped array
    
            answer=np.zeros((len(nus),tresolution),dtype=np.complex_)
            for nu_index,nu in enumerate(nus):
                #it is too much for my computer to handle doing this as an array because nu x tresolution x chiresolution is crazy big
                mint=mints[nu_index]
                ts=np.linspace(mint,1-1e-5,tresolution)
                t_dependent_part=(interpolated_Wg(chis[:,np.newaxis]*ts,FIELD2)+ts**(nu-2)*interpolated_Wg(chis[:,np.newaxis]/ts,FIELD2))#hopefully a chi *t shaped array
                
                total_integrand=t_independen_multiplicitave_factor[nu_index,:,np.newaxis]*t_dependent_part
    
                answer[nu_index]=integrate.simps(total_integrand,chis,axis=0)
          
                #takes around 0.3 seconds
            
            
            
    elif FIELD1[0]=="shear":
        if FIELD2[0]=="shear":

            FIELD1=SPECTRUM[0]
            FIELD2=SPECTRUM[1]
    
            chimin=cosmo_functions.comoving_distance(0.01) #what should this be?
            chimax=cosmo_functions.comoving_distance(lensing_kernels.boundaries_sources[redshift_bin+1])
            chis=np.linspace(chimin,chimax,chiresolution)
        
    
            t_independen_multiplicitave_factor=chis**(1-nus[:,np.newaxis])*interpolated_Wg(chis,FIELD1)#returns chi-shaped array
    
            answer=np.zeros((len(nus),tresolution),dtype=np.complex_)
            for nu_index,nu in enumerate(nus):
            #it is too much for my computer to handle doing this as an array because nu x tresolution x chiresolution is crazy big
                mint=mints[nu_index]
                ts=np.linspace(mint,1-1e-5,tresolution)
                t_dependent_part=(interpolated_Wg(chis[:,np.newaxis]*ts,FIELD2)+ts**(nu-2)*interpolated_Wg(chis[:,np.newaxis]/ts,FIELD2))#hopefully a chi *t shaped array
    
                total_integrand=t_independen_multiplicitave_factor[nu_index,:,np.newaxis]*t_dependent_part
            
                answer[nu_index]=integrate.simps(total_integrand,chis,axis=0)
          
            #takes around 0.3 seconds
        else:
            print("deal with shear/density")
            
    return answer



'''

below i find the minimum T to start integrating at for each (ell,nu) such that the function I_ellnu(t) is 
10^^-5 times smaller than its value at t=1.

see 1705.05022

'''
def zerot(t,ell,nu):
  #  return t+ell+nu-9
    return np.abs(i_ell_floats(ell,nu,t))-np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5)

def zerot5(t,ell,nu): #due to numerical instability around 0.7 i have to include this.. would be better to come up with some other way??
  #  return t+ell+nu-9
    return np.abs(i_ell_floats(ell,nu,t))-np.abs(i_ell_floats(ell,nu,0.999999999999999)*5e-5)
def Min_T(ell,nu):
    if(np.abs(i_ell_floats(ell,nu,0.6999))>np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5)):
        return optimize.brentq(zerot,0.01,0.98,args=(ell,nu))
    elif np.abs(i_ell_floats(ell,nu,0.7001))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5):
        return optimize.brentq(zerot,0.7001,0.98,args=(ell,nu))
    else:
        ts=np.linspace(0.71,0.99,29)
        for t in ts:
            if np.abs(i_ell_floats(ell,nu,t))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5):
                return optimize.brentq(zerot,t,0.98,args=(ell,nu))
        for t in ts:
            if np.abs(i_ell_floats(ell,nu,t))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*5e-5):
                return optimize.brentq(zerot5,t,0.98,args=(ell,nu))
        print("t problem")
        return None

 

 
def intIW(ells,nus,tresolution,chiresolution,SPECTRUM,experiment,nbins,test=False):
    
    #clgnu in the mathematica nb
    time1=time.time()

    
    answer=np.zeros((len(ells),len(nus)),dtype=np.complex_)
        
    ts=np.linspace(1e-5,1-1e-5,tresolution)

    fact2lessthan10=i_ell_tarray(ells[ells<11],nus,ts) #txnu shaped array
    fact1=intWgalaxy_lowell(ells[ells<11],nus,ts,chiresolution,SPECTRUM,nbins,experiment,test)#ellxtxnushaped array
       
    integrand=fact1*fact2lessthan10 
    answer[ells<11,:]= integrate.simps(integrand,ts,axis=1) 

    for ell_index, ell in enumerate(ells): 
        if ell>10:
            mints=interpolated_minT(ell,nus) 
          
            if np.max(mints)>(1-1e-5):
                print("t problem",mints[mints>1-1e-5])
            fact1=intWgalaxy_highell(ell,nus,tresolution,chiresolution,SPECTRUM,nbins,experiment,test) 
               #this is nu x tres
                #the problem is that when we take different mints then ts changes. 
                #some of this is not spectrum-specific which means we only need to do it ONCE, which is useful.
                #however we have to do fact1 separately for each spectrum.
                
            fact2=i_ell(ell,nus,tresolution) #want nu x tres 
           
            integrand=fact1*fact2
            for nu_index, nu in enumerate(nus):
                mint=mints[nu_index]
                ts=np.linspace(mint,1-1e-5,tresolution)
                
                answer[ell_index,nu_index]= integrate.simps(integrand[nu_index,:],ts)#remember to change t! ts is different for every nu         
    return answer

def Pkz(z,k,kh=False,Ph=False):
    #returns Plin(k with Plin in Mpc^3, k in Mpc)
    #should work for array of ks
    kp=1e-4
    h=cosmo_functions.H0/100
    
    answer=np.zeros(len(k))

    answer[k>kp]= cosmo_functions.Plin(z,k[k>kp]*(h**kh))*((h)**3)**Ph
    ns=0.97
    Pp=cosmo_functions.Plin(z,kp*(h**kh))*((h)**3)**Ph
    answer[k<kp]=Pp*(k[k<kp]/kp)**ns #ns is spectral index
    return answer

def Pk(k,kh=False,Ph=False,linear=True):
    #returns Plin(k with Plin in Mpc^3, k in Mpc)
    kp=1e-4
    h=cosmo_functions.H0/100
    answer=np.zeros(len(k))

    if linear:
        answer[k>=kp]=  cosmo_functions.Plin(0,k[k>=kp]*(h**kh))*((h)**3)**Ph
        ns=0.97
        Pp=cosmo_functions.Plin(0,kp*(h**kh))*((h)**3)**Ph
        answer[k<kp]= Pp*(k[k<kp]/kp)**ns #ns is spectral index
    else:
        answer[k>=kp]=  cosmo_functions.Pnonlin(0,k[k>=kp]*(h**kh))*((h)**3)**Ph
        ns=0.97
        Pp=cosmo_functions.Pnonlin(0,kp*(h**kh))*((h)**3)**Ph
        answer[k<kp]= Pp*(k[k<kp]/kp)**ns #ns is spectral index
        
    return answer

def to_transform(k):
    
    if test:
        return 1/k*Pk(k,True,True)*np.exp(-((k)/10)**2)
    return 1/(k)*Pk(k)*np.exp(-((k)/10)**2)#Pk(k,True,True)*np.exp(-((k)/10)**2)


cns,fns=coeffs(to_transform,200,1e-8,52,1.9)
ls=np.logspace(1,2,15)
ls=[int(l)for l in ls]
nus=[fns[8*i]for i in range(0,int(len(fns)/8))]
for i in range(0,len(nus)):
    if nus[i].imag==0:
        nus[i]=nus[i-1]
mints=np.zeros((len(ls),len(nus)))
for i,ell in enumerate(ls):
    for j,nu in enumerate(nus):
        mints[i,j]=Min_T(ell,nu)
        
jj=interp2d([nu.imag for nu in nus],np.array(ls),mints)  

def interpolated_minT(ell,nu):
    return jj(nu.imag,ell)                     
                                               

def _Cls_Exact_Wrapper(ells,tresolution,chiresolution,maxfreq,number_of_clustering_bins,bias,experiment=None,test=False):
    
    
        N_field=N_fields(number_of_clustering_bins)
        source_bins=10
        
        shear_fields=[["shear",i]for i in range(0,source_bins)]
        density_fields=[["density",i]for i in range(0,number_of_clustering_bins)]
                
        Cl=np.zeros((N_field,N_field,len(ells)))
           
        for i in range(0,source_bins):
            for j in range(0,source_bins):
                if i<=j:
                    SPECTRUM=[shear_fields[i],shear_fields[j]]
                    Cl[i+number_of_clustering_bins,j+number_of_clustering_bins,:]=Cl[j+number_of_clustering_bins,i+number_of_clustering_bins,:]=(ells*(ells+1))**2*Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment,bias,number_of_clustering_bins)
                              
        for i in range(0,number_of_clustering_bins):
            for j in range(0,number_of_clustering_bins):
                if i<=j:
                    SPECTRUM=[density_fields[i],density_fields[j]]
                    Cl[i,j,:]=Cl[j,i,:]=Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment,bias,number_of_clustering_bins)#galaxy galaxy lensing
                
        return Cl
                                               

def Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment,bias,NBINS,test=False):
    
    kmin=1e-8
    kmax=52
    Nmax=maxfreq
    
    cns,fns=coeffs(to_transform,Nmax,kmin,kmax,bias)
   
    xx=intIW(ells,fns[0:int((Nmax)/2)+1],tresolution,chiresolution,SPECTRUM,experiment,NBINS,test)
    
    ans=np.sum(cns[0:int(Nmax/2)]*xx[:,0:int(Nmax/2)],axis=1)/(2*np.pi**2)
    
    answer=ans+np.conj(ans)+cns[int(Nmax/2)]*xx[:,int(Nmax/2)]/(2*np.pi**2)

    return (answer.real)

Pnonlin=cosmo_functions.Pnonlin
Plin=cosmo_functions.Plin

source_bins=10


def _Cls_Limber_Wrapper(spec,ls,resolution,number_of_clustering_bins,experiment=None,test=False):
    
    if  spec[0]=="shear" and not test:
        shears=["shear_"+str(i)for i in range(0,source_bins)]
    
        shear_powerspectra=np.zeros((source_bins,source_bins,len(ls)))
    
        Ws=np.zeros((source_bins,resolution)) #want to evaluate the redshift kernel FOR EACH SOURCE BIN AT EACH Z SAMPLE
                                              #set up the Ws by evaluating at each z      
        zs=np.linspace(0.01,4,resolution)
        chis=cosmo_functions.comoving_distance(zs)
        for i in range(0,source_bins):
                 Ws[i,:]=lensing_kernels.W(shears[i],chis,number_of_clustering_bins) # /cosmo_functions.comoving_distance(chis) **2
                 
                 
        for i in range(0,source_bins):
            Ws1=Ws[i]
            for j in range(0,source_bins):
                Ws2=Ws[j]
                if j>=i:
                    shear_powerspectra[i,j]=shear_powerspectra[j,i]=Cls_limber(ls,Ws1,Ws2,Plin,chis,zs)
                        
    
        return shear_powerspectra
    elif spec[0]=="clustering":
        
    
        N_field=N_fields(number_of_clustering_bins)
        bini_powerspectra=np.zeros((N_field,len(ls)))

        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        redshift_bin=spec[1]
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]
        
        

        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))
        
        if test:
            zmin=cosmo_functions.redshift(3000-5*300)
            zmax=cosmo_functions.redshift(3000+5*300)
            galaxy_bias=1

        zs=np.linspace(zmin,zmax,resolution)
        chis=cosmo_functions.comoving_distance(zs)
        W_clustering=np.zeros(len(chis))
        if not test:
            normalisation=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)
        if test:
            normalisation = 1
            
        if not test:
        
            W_clustering=galaxy_bias*lensing_kernels.W("density_"+str(redshift_bin+1),chis,number_of_clustering_bins,experiment)/normalisation
        
        
        if test:
            W_clustering=Wg(chis, "a",number_of_clustering_bins,"magic",True)
            
       
        
        Ws=np.zeros((source_bins,resolution))
        
        if not test:
            for i in range(0,source_bins):
                    if(zmin<lensing_kernels.boundaries_sources[i+1])  :  #want the lenses to be BEHIND the clustering galaxies
                        Ws[i,:]=lensing_kernels.W("shear_"+str(i),chis,number_of_clustering_bins)# /cosmo_functions.comoving_distance**2
                    
                    bini_powerspectra[number_of_clustering_bins+i,:]=Cls_limber(ls,W_clustering,Ws[i],Plin,chis,zs)
        
        if not test:
        
            bini_powerspectra[redshift_bin,:]=Cls_limber(ls,W_clustering,W_clustering,Plin,chis,zs)
        else:
            for ell_index,ell in enumerate(ls):
                integrand=1/chis**2*(W_clustering*W_clustering)*Pk((ell+1/2)/chis,True,True) 
                bini_powerspectra[redshift_bin,ell_index]=integrate.simps(integrand,chis)
        
        return bini_powerspectra
    else:
        print("spec problem; spec is",spec)
        return 


def Cls_limber(ls,Ws1,Ws2,matter_pk,chis,zs):#specs tells you which power spectrum you want
   
    #ls is the l-values you will compute ats
    #Ws1 and Ws2 are the kernels you are integrating
   
    
    Cls=[]
   
    zs=np.array(zs)
    chis=np.array(chis)  
    int1= Ws1 * Ws2 /chis**2 
   
    for l in ls:
        
        int2=np.array([matter_pk(zs[k],(l+1/2)/chis[k]) for k in range(0,len(chis))]) #would be cool if i could get this to work for arrays?
        integrand=int1*int2
        Cls.append(integrate.simps(integrand,chis))
    return np.array(Cls)

print("integrating_cls_bessel imported")
