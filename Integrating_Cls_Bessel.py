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
import matplotlib.pyplot as plt



from scipy.interpolate import interp1d,interp2d
from scipy import integrate
from scipy import optimize
from scipy import special

import mpmath

import cosmo_functions

import lensing_kernels


import time

H0=cosmo_functions.H0
c=cosmo_functions.c
omegam=cosmo_functions.omegam

#lensing_bins=config.lensing_bins


def coeffs(function,Nmax,kmin,kmax,bias):
    
    kappamin,kappamax=np.log(kmin),np.log(kmax)
    
    ints=np.linspace(-int(Nmax/2),int(Nmax/2),Nmax+1-Nmax%2)

    period=kappamax-kappamin
   
    
    kappas=kappamin+np.linspace(0,period,Nmax+1-Nmax%2)
    samples=[function(np.exp(kappa))*(np.exp(kappa)/np.sqrt(kmin*kmax))**bias for kappa in kappas]
    
   # plt.plot(np.exp(kappas),samples,'o')
   # plt.xscale('log')
   # plt.show()
  #  looks good.
    cns=np.zeros(len(ints),dtype=np.complex_)
    fns=np.zeros(len(ints),dtype=np.complex_)
  
    for i in range(0,len(cns)):
        fns[i]=2*np.pi*(0+1j)*ints[i]/period-bias

        cns[i]=1/Nmax*np.sum(samples*np.exp(-(2*np.pi*(0+1j)*ints[i]*ints)/Nmax)) *1/(np.sqrt(kmin*kmax))**fns[i]
        
    if Nmax%2==0:
        cns[0]=cns[0]/2
        cns[-1]=cns[-1]/2
    return cns,fns

def logf_transform(function,x,Nmax,xmin,xmax,bias):
    cns,fns=coeffs(function,Nmax,xmin,xmax,bias)
   # print(cns/(np.sqrt(xmin*xmax))**(fns),fns)
    return np.sum([cns[i]*((x/1))**(fns[i]) for i in range(0,len(cns))])

def hypf(a,b,c,d):
    return mpmath.hyp2f1(a,b,c,d)

def hypfnearone(ell,nu,t):
    ##equatino B5 of 1705.05022 as equation b6 did not work for me
     ttilde=-(1-t**2)**2/(4*t**2)
     return special.gamma(ell+3/2)*t**(-ell-nu/2)*(special.gamma(nu-2)*t**(nu-2)*(1-t**2)**(2-nu)/(special.gamma((nu-1)/2)*special.gamma(ell+nu/2))*hypf((2*ell-nu+4)/4,(-2*ell-nu+2)/4,2-nu/2,ttilde)+special.gamma(2-nu)/(special.gamma(3/2-nu/2)*special.gamma(ell-nu/2+2))*hypf((2*ell+nu)/4,(-2*ell+nu-2)/4,nu/2,ttilde))


'''
would like to turn the bottom three into one function.

'''
    
def i_ell_floats(ell,nu,t):
    
    #would like this to take in an ARRAY of ts
    tstar=0.7
    if t<tstar:
        return 2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*t**ell* hypf((nu-1)/2,ell+nu/2,ell+3/2,t**2)
    else:
       
        return 2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*t**ell*hypfnearone(ell,nu,t)

def i_ell(ell,nus,ts):
    
    #would like this to take in an ARRAY of ts and return some array
    
    answer=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    hypfs=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    tstar=0.7
    
    
    for i,t in enumerate(ts):
        
        #would be good to get hypf working for arrays?
        if t<tstar:
            for j,nu in enumerate(nus):
                hypfs[i,j]=hypf((nus[j]-1)/2,ell+nus[j]/2,ell+3/2,t**2)
        else:
            for j,nu in enumerate(nus):
                hypfs[i,j]=hypfnearone(ell,nus[j],t)
    
    answer[ts<tstar]= 2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2))* hypfs[ts<tstar]*ts[ts<tstar,np.newaxis]**ell
    answer[ts>tstar]=2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2))*hypfs[ts>tstar]*ts[ts>tstar,np.newaxis]**ell
    
    return answer

def i_ellnew2(ell,nus,ts):
    
    #would like this to take in an ARRAY of ts and return some array
    
    answer=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    hypfs=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    tstar=0.7
    
    
    for i,t in enumerate(ts):
        
        #would be good to get hypf working for arrays?
        if t<tstar:
            for j,nu in enumerate(nus):
                hypfs[i,j]=hypf((nus[j]-1)/2,ell+nus[j]/2,ell+3/2,t**2)
        else:
            for j,nu in enumerate(nus):
                hypfs[i,j]=hypfnearone(ell,nus[j],t)
    
    answer[ts<tstar]= 2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2))* hypfs[ts<tstar]*ts[ts<tstar,np.newaxis]**ell
    answer[ts>tstar]=2**(nus-1)*np.pi**2*special.gamma(ell+nus/2)/(special.gamma((3-nus)/2)*special.gamma(ell+3/2))*hypfs[ts>tstar]*ts[ts>tstar,np.newaxis]**ell
    
    return answer




def i_ellnew(ell,nu,ts):
    
    #would like this to take in an ARRAY of ts and return some array
    
    answer=np.zeros(len(ts),dtype=np.complex_)
    hypfs=np.zeros(len(ts),dtype=np.complex_)
    tstar=0.7
    
    
    for i,t in enumerate(ts):
        #would be good to get hypf working for arrays?
        if t<tstar:
            hypfs[i]=hypf((nu-1)/2,ell+nu/2,ell+3/2,t**2)
        else:
            hypfs[i]=hypfnearone(ell,nu,t)
    
    answer[ts<tstar]= 2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*ts[ts<tstar]**ell* hypfs[ts<tstar]
    answer[ts>tstar]=2**(nu-1)*np.pi**2*special.gamma(ell+nu/2)/(special.gamma((3-nu)/2)*special.gamma(ell+3/2))*ts[ts>tstar]**ell*hypfs[ts>tstar]
    
    return answer


'''
turn the above three into one function.
'''

def N_fields(clustering_bins):
    return    clustering_bins+source_bins


def bin_boundaries(clustering_bins):
    return np.linspace(0.2,1,clustering_bins+1)


def Wg(chis, FIELD,number_of_clustering_bins,experiment="magic"):
    #FIELD SHOULD BE A LIST ["TYPE",BIN_NUMBER]
    #WHERE "TYPE" IS EITHER "density" or "shear" 
    #and BIN_NUMBER is the relevant bin number starting indexing at 0. 
    #if the field is "density" there should be a third item in the list, EXPERIMENT, which should be "LSST" or "magic"
    
    #remember the power spectrum we transform is P(z=0,k) so we must multiply by the growth factor normalized to 1 at z=0
    
     if FIELD[0]=="density":
        
        redshift_bin=FIELD[1]
        


        z_bin_boundaries=bin_boundaries(number_of_clustering_bins)
        zmin=z_bin_boundaries[redshift_bin]
        zmax=z_bin_boundaries[redshift_bin+1]
       # print(zmin,zmax)
        galaxy_bias=0.95/cosmo_functions.D_growth_norm1_z0(cosmo_functions.z2a((zmin+zmax)/2))
        
        zs=np.linspace(zmin,zmax,100)#is this enough resolution
        
        normalisation=integrate.simps([lensing_kernels.dnilensdz(z,experiment)for z in zs],zs)
       # print(normalisation)
        W=galaxy_bias*lensing_kernels.W("density_"+str(redshift_bin+1),chis,experiment)/normalisation
        
        
     if FIELD[0]=="shear":
        redshift_bin=FIELD[1]
        #come back to this.
        W=lensing_kernels.W("shear_"+str(redshift_bin),chis)
        
     return W*cosmo_functions.D_growth_chi_normz0(chis) #include growth factor in definition of Wg.
 
def firstderiv_Wg(chi,FIELD,number_of_clustering_bins):
    
    deltachi=chi*0.000001
    
    
    return (Wg(chi+deltachi,FIELD,number_of_clustering_bins)-Wg(chi-deltachi,FIELD,number_of_clustering_bins))/(2*deltachi)

def secondderiv_wg(chi,FIELD,number_of_clustering_bins):
    
    deltachi=chi*0.000001
    
    
    return (Wg(chi-deltachi,FIELD,number_of_clustering_bins)+Wg(chi+deltachi,FIELD,number_of_clustering_bins)-2*Wg(chi,FIELD,number_of_clustering_bins))/(deltachi**2)

    
def first_part(chi,FIELD,number_of_clustering_bins):
     return -secondderiv_wg(chi,FIELD,number_of_clustering_bins)+2/chi*firstderiv_Wg(chi,FIELD,number_of_clustering_bins)
 
    
chis=np.linspace(2e-6,13000,1000) #is this enough chi-sampling? I should change this.

number_of_clustering_bins=4

W_interp=interp1d(chis,Wg(chis,["density",0,"magic"],number_of_clustering_bins))



W_interp_shears=[interp1d(chis,Wg(chis,["shear",i,"magic"],number_of_clustering_bins))for i in range(0,10)]


interp4_clustering=interp1d(chis,first_part(chis,["density",0,"magic"],number_of_clustering_bins))

interp4_shears=[interp1d(chis,first_part(chis,["shear",i,"magic"],number_of_clustering_bins))for i in range(0,10)]

'''
plt.plot(chis,Wg(chis,["density",0,"magic"],number_of_clustering_bins))
plt.show()
plt.plot(chis,first_part(chis,["density",0,"magic"],number_of_clustering_bins))
plt.show()
 '''

def interpolated_Wg(chis,FIELD):
    #I WOULD LIKE THIS TO RETURN THE CORRECT INTERPOLATING FUNCTION FOR THE WINDOW FUNCTION WG.
    #REMEMBER WG SHOULD BE MULTIPLIED BY THE GROWTH FACTOR NORMALIZED TO 1 AT Z=0 BECAUSE THE
    #POWER SPECTRUM WE ARE USING IS P(Z=0,K).
#    print("FIEL DIS",FIELD)
    if FIELD[0]=="density":
          
        return W_interp(chis)
    else:
     #   print("field is",FIELD)
      #  print(W_interp_shears[FIELD[1]](chis))
        return W_interp_shears[FIELD[1]](chis)

def interp_dlwg_first(chis,FIELD):
  #  print("about to find interpolation")
   # print("field is",FIELD)
    if FIELD[0]=="density":
    #    print("clustering answer",interp4_clustering(chis))
    #I WOULD LIKE THIS TO RETURN THE CORRECT INTERPOLATING FUNCTION FOR -D^2D/CHI^2 + 2/CHI D/DCHI BASED ON THE APPROPRIATE FIELD.
        return interp4_clustering(chis)
    elif FIELD[0]=="shear":
     #   print("in shear part",FIELD)
        return interp4_shears[FIELD[1]](chis)


def Dl_wg(chi,ell,FIELD,experiment):
    
    #interp4 should be -d^2d/chi^2 +2/chi d/dchi I think. double check this.
    ans=np.zeros(chi.shape)
#    print(ans.shape)
##    print(FIELD)
 #   print("one",interp_dlwg_first(chi[chi<13000],FIELD))
    
   # print("two")
#    print((interp_dlwg_first(chi[chi<13000],FIELD)))
 #   print("okay")
  #  print("two",FIELD,chi[chi<13000],(interp_dlwg_first(chi[chi<13000],FIELD)))
    ans[chi<13000]=interp_dlwg_first(chi[chi<13000],FIELD)+(ell*(ell+1)-2)/chi[chi<13000]**2*interpolated_Wg(chi[chi<13000],FIELD)
    
   # plt.plot(chi[chi<8000],ans[chi<8000])
   # plt.show()
    
    return ans



def intWgalaxynew2(ell,nus,ts,chiresolution,SPECTRUM,experiment):
   # print("spec is",SPECTRUM)
    FIELD1=SPECTRUM[0]
    FIELD2=SPECTRUM[1]
    
    #now i would like this to work for an array of nus as well as ts.
    
    answer=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    #i would like this to take in an array of ts and give me back some array with one element for every t.
    
    ##MAKE BOUNDARIES
    chimin=cosmo_functions.comoving_distance(0.2)
    chimax=cosmo_functions.comoving_distance(0.4)
    chis=np.linspace(chimin,chimax,chiresolution)
   # print("chis",chis)
    t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array
                                                                                                                                         #would like a chi-times-nu-times-t shaped array
  #  print("tin",t_independen_multiplicitave_factor)
    first=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2,experiment)) #chi times t
    
#    print("first",first)
    second=Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2,experiment)#chi times t
#    print("second",second)
                                                                                                                      #would like a chi-times-nu-times-t shaped array
    secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
    
    firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
    
    total=(Dl_wg(chis,ell,FIELD1,experiment)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
     
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
   # print(total_integrand.shape)
    for i in range(0,len(ts)):
        for j in range(0,len(nus)):
            answer[i,j]=integrate.simps(total_integrand[:,i,j],chis)
  #  print("answer",answer)
    return answer #a t -times nu shaped array.

def intWgalaxynew(ell,nu,ts,chiresolution,SPECTRUM,experiment):
    
    
    #this is for a complex nu (as opposed to an array)
    FIELD1=SPECTRUM[0]
    FIELD2=SPECTRUM[1]
    answer=np.zeros(len(ts),dtype=np.complex_)
    #i would like this to take in an array of ts and give me back some array with one element for every t.
    chimin=cosmo_functions.comoving_distance(0.2)
    chimax=cosmo_functions.comoving_distance(0.4)
    chis=np.linspace(chimin,chimax,chiresolution)
    
    t_independen_multiplicitave_factor=chis**(1-nu)*Dl_wg(chis,ell,FIELD1,experiment)#returns chi-shaped array
    
  #  integrand=chis[np.newaxis]**(1-nu)*Dl_wg(chis,sigmax1,chi1av,ell)*(Dl_wg(chis*t,sigmax2,chi2av,ell)+t**(nu-2)*Dl_wg(chis/t,sigmax2,chi2av,ell))
    
    t_dependent_part=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2,experiment)+ts**(nu-2)*Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2,experiment))#hopefully a chi *t shaped array
    
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis]*t_dependent_part
    
    for i in range(0,len(ts)):
        answer[i]=integrate.simps(total_integrand[:,i],chis)
    
    return answer

'''
def intWgalaxynew(ell,nu,ts,chiresolution,SPECTRUM,experiment):

    
    if SPECTRUM[0]=="clustering":
        FIELD1=["density",SPECTRUM[1],experiment]
        FIELD2=FIELD1
    
    #this only works for complex float nu
    answer=np.zeros(len(ts),dtype=np.complex_)

    chimin=0.4#fill in
    chimax=0.6#fill in
    chis=np.linspace(chimin,chimax,chiresolution)
    
    t_independen_multiplicitave_factor=chis**(1-nu)*Dl_wg(chis,ell,FIELD1)#returns chi-shaped array
    
    
    t_dependent_part=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2)+ts**(nu-2)*Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2))#hopefully a chi *t shaped array
    
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis]*t_dependent_part
    
    for i in range(0,len(ts)):
        answer[i]=integrate.simps(total_integrand[:,i],chis)
    
    return answer





def intWgalaxy(ell,nus,ts,chiresolution,SPECTRUM,experiment):
     
    if SPECTRUM[0]=="clustering":
        FIELD1=["density",SPECTRUM[1],experiment]
        FIELD2=FIELD1
    
    #this works for nus an array as well as ts an array   
    
    answer=np.zeros((len(ts),len(nus)),dtype=np.complex_)

    chimin=0.4#fill in
    chimax=0.6#fill in
    chis=np.linspace(chimin,chimax,chiresolution)
    
     
    t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array
    
                                                                                                                                        #would like a chi-times-nu-times-t shaped array
    first=(Dl_wg(chis[:,np.newaxis]*ts,ell,FIELD2)) #chi times t
    
    second=Dl_wg(chis[:,np.newaxis]/ts,ell,FIELD2)#chi times t
                                                                                                                                        #would like a chi-times-nu-times-t shaped array
    secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
    
    firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
    
    total=(Dl_wg(chis,ell,FIELD1)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
     
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
 #   print(total_integrand.shape)
    for i in range(0,len(ts)):
        for j in range(0,len(nus)):
            answer[i,j]=integrate.simps(total_integrand[:,i,j],chis)
    
    return answer #a t -times nu shaped array.

'''
def zerot(t,ell,nu):
  #  return t+ell+nu-9
    return np.abs(i_ell_floats(ell,nu,t))-np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5)
def Min_T(ell,nu):
    if(np.abs(i_ell_floats(ell,nu,0.6999))>np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5)):
       # print("in")
        return optimize.brentq(zerot,0.01,0.98,args=(ell,nu))
    elif np.abs(i_ell_floats(ell,nu,0.7001))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5):
       # print("in2")
        return optimize.brentq(zerot,0.7001,0.98,args=(ell,nu))
    else:
        ts=np.linspace(0.71,0.99,29)
        for t in ts:
            if np.abs(i_ell_floats(ell,nu,t))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*1e-5):
                return optimize.brentq(zerot,t,0.98,args=(ell,nu))
        for t in ts:
            if np.abs(i_ell_floats(ell,nu,t))<np.abs(i_ell_floats(ell,nu,0.999999999999999)*5e-5):
                return optimize.brentq(zerot,t,0.98,args=(ell,nu))
        print("t problem")
        return None

 


def intIW(ell,nus,tresolution,chiresolution,SPECTRUM,experiment):
    #clgnu in the mathematica nb

    answer=np.zeros(len(nus),dtype=np.complex_)

  
    if(ell<11):
        mint=1e-5
        ts=np.linspace(mint,1-mint,tresolution)#some sampling near 0
    
      
        fact1=intWgalaxynew2(ell,nus,ts,chiresolution,SPECTRUM,experiment)#txnushaped array
       # print("fact1",fact1[:,0])

        fact2=i_ellnew2(ell,nus,ts) #txnu shaped array
      #  print("fact2",fact2[:,0])

        integrand=fact1*fact2 #fact1 and fact2 should be t-times nu shaped arrays


        for i in range(0,len(answer)):
            answer[i]= integrate.simps(integrand[:,i],ts) #this should be a nu-shaped array
       
        
       # print("integrand",integrand[:,0])

        return answer #reminder, answer has to be a nu-shaped array.
    else:
        mints=interpolated_minT(ell,nus)
        
        if mints.any()>(1-1e-5):
            print("t problem",mints[mints>1-1e-5])
            
        for i in range(0,len(nus)):
            mint=mints[i]
            ts=np.linspace(mint,1-1e-5,tresolution)
   
            fact1=intWgalaxynew(ell,nus[i],ts,chiresolution,SPECTRUM,experiment)
            fact2=i_ellnew(ell,nus[i],ts) 
      
            integrand=fact1*fact2

            answer[i]= integrate.simps(integrand,ts)
        return answer
'''
ks_read=[]
pks_read=[]
ps=open("PCAMBz0.txt",'r')
lines=ps.readlines()
for i in lines:
    k=i.split()[0]
    #print(float(k.split('E')[0]))
    k=float(k.split('E')[0])*10**float(k.split('E')[1])
    pk=i.split()[1]
  #  print(pk)
    pk=float(pk.split('E')[0])*10**float(pk.split('E')[1])
    ks_read.append(k)
    pks_read.append(pk)
    
ps.close()

'''





def Pkz(z,k,kh=False,Ph=False):
    #returns Plin(k with Plin in Mpc^3, k in Mpc)
    kp=1e-4
    h=cosmo_functions.H0/100

    if k>kp:
        return cosmo_functions.Plin(z,k*(h**kh))*((h)**3)**Ph
    else:
        ns=0.97
        Pp=cosmo_functions.Plin(z,kp*(h**kh))*((h)**3)**Ph
        return Pp*(k/kp)**ns #ns is spectral index

def Pk(k,kh=False,Ph=False,linear=True):
    #returns Plin(k with Plin in Mpc^3, k in Mpc)
    kp=1e-4
    h=cosmo_functions.H0/100
    if linear:
        if k>kp:
            return cosmo_functions.Plin(0,k*(h**kh))*((h)**3)**Ph
        else:
            ns=0.97
            Pp=cosmo_functions.Plin(0,kp*(h**kh))*((h)**3)**Ph
            return Pp*(k/kp)**ns #ns is spectral index
    else:
         if k>kp:
            return cosmo_functions.Pnonlin(0,k*(h**kh))*((h)**3)**Ph
         else:
            ns=0.97
            Pp=cosmo_functions.Pnonlin(0,kp*(h**kh))*((h)**3)**Ph
            return Pp*(k/kp)**ns #ns is spectral index
        
    
def to_transform(k):
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
        #print(j)
        mints[i,j]=Min_T(ell,nu)
        
jj=interp2d([nu.imag for nu in nus],np.array(ls),mints)  

def interpolated_minT(ell,nu):
    return jj(nu.imag,ell)                     
                                               

def _Cls_Exact_Wrapper(FIELD,ells,tresolution,chiresolution,maxfreq,number_of_clustering_bins,experiment=None):
    
    
    #FIELD SHOULD BE LIKE ["DENSITY", i] OR ["SHEAR",i] WITH i THE REDSHFIT BIN
    
    if FIELD[0]=="shear":
        '''
        fill in this part!!
        '''
        '''
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
        '''
        return None
    elif FIELD[0]=="density":
        
        N_field=N_fields(number_of_clustering_bins)

        bini_powerspectra=np.zeros((N_field,len(ells)))
        
        redshift_bin=FIELD[1]

        shear_fields=[["shear",i ]for i in range(0,10)]
        
        for i in range(0,source_bins):
            #COME BACK TO THIS
                   SPECTRUM=[FIELD,shear_fields[i]]
                   bini_powerspectra[number_of_clustering_bins+i,:]=Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment)#galaxy galaxy lensing
                #bini_powerspectra[number_of_clustering_bins+i,:]=Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq)
        
        
        SPECTRUM=[FIELD,FIELD]
        bini_powerspectra[redshift_bin,:]=Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment) #AUTO POWER SPECTRUM
        
        return bini_powerspectra
    else:
        print("spec problem; spec is",FIELD)
        return 
                                               

def Cls_Exact(ells,SPECTRUM,tresolution,chiresolution,maxfreq,experiment):
    bias=1.9   #BIAS FOR FOURIER TRANSFORMING THE POWER SPECTRUM
    t1=time.time()
    
    kmin=1e-8
    kmax=52
    Nmax=maxfreq
    
    cns,fns=coeffs(to_transform,Nmax,kmin,kmax,bias)
    
    answer=[]
    for ell in ells:
       # print(cns[0:10])
        print(ell)
       # print(intIW(ell,fns[0:10],tresolution,chiresolution,SPECTRUM,experiment))
        ans=np.sum(cns*intIW(ell,fns,tresolution,chiresolution,SPECTRUM,experiment))/(2*np.pi**2)
        answer.append(ans)
   # print(answer,"in ",time.time()-t1,"seconds")

    return ([ans.real for ans in answer])

'''
def limbercl(ell,chiresolution,spec):
    
    
    chis=np.linspace(0.01,chimax,chiresolution)
    
    integrand=[1/chi**2*(Wg(chi,sigmax1,chi1av)*Wg(chi,sigmax2,chi2av))*Pk((ell+1/2)/chi,True,True) for chi in chis]
    return integrate.simps(integrand,chis)#,chis,integrand
'''
Pnonlin=cosmo_functions.Pnonlin
source_bins=10


def N_fields(clustering_bins):
    return    clustering_bins+source_bins



def bin_boundaries(clustering_bins):
    return np.linspace(0.2,1,clustering_bins+1)


def _Cls_Limber_Wrapper(spec,ls,resolution,number_of_clustering_bins,experiment=None):
    
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



def Cls_limber(ls,Ws1,Ws2,matter_pk,chis,zs):#specs tells you which power spectrum you want
   
    #ls is the l-values you will compute ats
    #Ws1 and Ws2 are the kernels you are integrating
   
    #zs=redshift(chis)
    
    Cls=[]
    Ws1=np.array(Ws1)  #maybe pass in arrays?? compute these as arrays?
    Ws2=np.array(Ws2)
    zs=np.array(zs)
    chis=np.array(chis)  
    int1= Ws1 * Ws2 /chis**2 
    #plt.plot(chis,Ws1)
   # plt.show()
    for l in ls:
        #integrand=[Ws1[k]*Ws2[k]*matter_pk(zs[k],(l+1/2)/chis[k])/chis[k]**2 for k in range(0,len(chis))]
        
        int2=np.array([matter_pk(zs[k],(l+1/2)/chis[k]) for k in range(0,len(chis))]) #would be cool if i could get this to work for arrays?
        #int2=matter_pk(zs,(l+1/2)/chis).diagonal()
        integrand=int1*int2
        Cls.append(integrate.simps(integrand,chis))
        #print("plotting",ls,l)
        #plt.plot(chis,integrand)
     #   3plt.show()
    return np.array(Cls)

print("integrating_cls_bessel imported")
