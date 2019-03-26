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

import sys
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




def gaussianwindow(chi,sigma,chiav):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(chi-chiav)**2/(2*sigma**2))

def Wg(chi,sigma,chiav):
    if(chi<8000):
      #  print(chi)
        return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(chi-chiav)**2/(2*sigma**2))*cosmo_functions.D_growth_chih_normz0(chi)
    else:
        return 0

def firstderiv_wg(chi,sigma,chiav):
    #####
    if(chi<8000):
        dfirst=-1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(chi-chiav)**2/(2*sigma**2))*(chi-chiav)/sigma**2
        dsecond=cosmo_functions.dgrowthdchi(chi)
        return dfirst*cosmo_functions.D_growth_chih_normz0(chi)+dsecond*gaussianwindow(chi,sigma,chiav)
    return 0

def second_deriv_wg(chi,sigma,chiav):
    if chi<8000:
        first= gaussianwindow(chi,sigma,chiav)/sigma**4*(-sigma**2+(chi-chiav)**2)*cosmo_functions.D_growth_chih_normz0(chi)
        second=2*(-1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(chi-chiav)**2/(2*sigma**2))*(chi-chiav)/sigma**2)*cosmo_functions.dgrowthdchi(chi)
        third=gaussianwindow(chi,sigma,chiav)*cosmo_functions.d2growthdchi(chi)
        return first+second+third
    return 0
chires=10000
chis=np.linspace(0.01,8000,chires)
interp1=interp1d(chis,[Wg(chi,300,3000)for chi in chis]) #genearalise this.

def interpolated_Wg(chi,sigma,chiav):
    return interp1(chi)
interp2=interp1d(chis,[firstderiv_wg(chi,300,3000)for chi in chis])
def interpolated_deriv(chi,sigma,chiav):
    return interp2(chi)
interp3=interp1d(chis,[second_deriv_wg(chi,300,3000)for chi in chis])
def interpolated_secondderiv(chi,sigma,chiav):
    return interp3(chi)




def first_funct(chi,sigma,chiav):
    return-interpolated_secondderiv(chi,sigma,chiav)+2/chi*interpolated_deriv(chi,sigma,chiav)
    
interp4=interp1d(chis,[first_funct(chi,300,3000) for chi in chis])
def interp_first(chi):
    return interp4(chi)
print("done interpolating")


def Dl_wg(chi,sigma,chiav,ell):
    
    
    ans=np.zeros(chi.shape)
    
    ans[chi<8000]=interp4(chi[chi<8000])+(ell*(ell+1)-2)/chi[chi<8000]**2* interpolated_Wg(chi[chi<8000],sigma,chiav)
    return ans

def intWgalaxynew(ell,nu,chi1av,sigmax1,chi2av,sigmax2,ts,chiresolution):
    #this is for a complex nu (as opposed to an array)
    answer=np.zeros(len(ts),dtype=np.complex_)
    #i would like this to take in an array of ts and give me back some array with one element for every t.
    chimin=chi2av-5*sigmax2
    chimax=chi2av+5*sigmax1
    chis=np.linspace(chimin,chimax,chiresolution)
    
    t_independen_multiplicitave_factor=chis**(1-nu)*Dl_wg(chis,sigmax1,chi1av,ell)#returns chi-shaped array
    
  #  integrand=chis[np.newaxis]**(1-nu)*Dl_wg(chis,sigmax1,chi1av,ell)*(Dl_wg(chis*t,sigmax2,chi2av,ell)+t**(nu-2)*Dl_wg(chis/t,sigmax2,chi2av,ell))
    
    t_dependent_part=(Dl_wg(chis[:,np.newaxis]*ts,sigmax2,chi2av,ell)+ts**(nu-2)*Dl_wg(chis[:,np.newaxis]/ts,sigmax2,chi2av,ell))#hopefully a chi *t shaped array
    
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis]*t_dependent_part
    
    for i in range(0,len(ts)):
        answer[i]=integrate.simps(total_integrand[:,i],chis)
    
    return answer





def intWgalaxy(ell,nus,chi1av,sigmax1,chi2av,sigmax2,ts,chiresolution):
    
    #now i would like this to work for an array of nus as well as ts.
    
    answer=np.zeros((len(ts),len(nus)),dtype=np.complex_)
    #i would like this to take in an array of ts and give me back some array with one element for every t.
    chimin=chi2av-5*sigmax2
    chimax=chi2av+5*sigmax1
    chis=np.linspace(chimin,chimax,chiresolution)
    
  #  t_independen_multiplicitave_factor=chis**(1-nu)*Dl_wgnew(chis,sigmax1,chi1av,ell)#returns chi-shaped array
                                                                                     #would like a chi-times-new shaped array
    
    t_independen_multiplicitave_factor=chis[:,np.newaxis]**(1-nus)#returns chi times nu array
    
  #  integrand=chis[np.newaxis]**(1-nu)*Dl_wg(chis,sigmax1,chi1av,ell)*(Dl_wg(chis*t,sigmax2,chi2av,ell)+t**(nu-2)*Dl_wg(chis/t,sigmax2,chi2av,ell))
    
  #  t_dependent_part=(Dl_wgnew(chis[:,np.newaxis]*ts,sigmax2,chi2av,ell)+ts**(nus-2)*Dl_wgnew(chis[:,np.newaxis]/ts,sigmax2,chi2av,ell))#hopefully a chi *t shaped array
                                                                                                                                        #would like a chi-times-nu-times-t shaped array
    first=(Dl_wg(chis[:,np.newaxis]*ts,sigmax2,chi2av,ell)) #chi times t
    
    second=Dl_wg(chis[:,np.newaxis]/ts,sigmax2,chi2av,ell)#chi times t
                                                                                                                                        #would like a chi-times-nu-times-t shaped array
    secondmultipliedbytfactor=ts[np.newaxis,:,np.newaxis]**(nus-2)*second[:,:,np.newaxis]#chi times t times nu
    
    firstplussecond=first[:,:,np.newaxis]+secondmultipliedbytfactor #chi times t times nu
    
    total=(Dl_wg(chis,sigmax1,chi1av,ell)[:,np.newaxis,np.newaxis])*firstplussecond #chi times t times nu
     
    total_integrand=t_independen_multiplicitave_factor[:,np.newaxis,:]*total #chi times t times nu
    print(total_integrand.shape)
    for i in range(0,len(ts)):
        for j in range(0,len(nus)):
            answer[i,j]=integrate.simps(total_integrand[:,i,j],chis)
    
    return answer #a t -times nu shaped array.


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
        print("t problem")
        return None

 


def intIW(ell,nus,chi1av,sigmax1,chi2av,sigmax2,tresolution,chiresolution):
    #clgnu in the mathematica nb
    #would like to adapt this to take in an ARRAY of frequencies and give out an array with one element per frequency.
    answer=np.zeros(len(nus),dtype=np.complex_)

    #obviously i would like to optimize all this.
    
    #what about intwgalaxy and i_ell taking in an ARRAY of ts?
    #now it would be good for intwgalaxy and fact2 to take in the whole array of nus.
    
    if(ell<11):
        mint=1e-5
        ts=np.linspace(mint,1-mint,tresolution)#some sampling near 0
    
      
        fact1=intWgalaxy(ell,nus,chi1av,sigmax1,chi2av,sigmax2,ts,chiresolution)#txnushaped array
            
        fact2=i_ell(ell,nus,ts) #txnu shaped array

        integrand=fact1*fact2 #fact1 and fact2 should be t-times nu shaped arrays
            #print(integrand)
        print(integrand.shape)
        for i in range(0,len(answer)):
           # print(integrand[i].shape)
            answer[i]= integrate.simps(integrand[:,i],ts) #this should be a nu-shaped array
        return answer #reminder, answer has to be a nu-shaped array.
    else:
        mints=interpolated_minT(ell,nus)
        if mints.any()>(1-1e-5):
            print("t problem",mints[mints>1-1e-5])
            
        for i in range(0,len(nus)):
            mint=mints[i]
            ts=np.linspace(mint,1-1e-5,tresolution)
   
            fact1=intWgalaxynew(ell,nus[i],chi1av,sigmax1,chi2av,sigmax2,ts,chiresolution)
            fact2=i_ell(ell,nus[i],ts) 
      
            integrand=fact1*fact2

            answer[i]= integrate.simps(integrand,ts)
        return answer

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

ns=0.97
lowks=np.logspace(-10,np.log10(ks_read[0]),100)
lowpks=[pks_read[0]*(lowk/ks_read[0])**ns for lowk in lowks]
newks=[9.999999999999982e-09, 1.1189846716375937e-08, 1.252126695359891e-08, 1.4011105790559556e-08, 1.5678212612328844e-08, 1.7543679591871202e-08, 1.9631108547425115e-08, 2.1966909551822488e-08, 2.4580635071738764e-08, 2.750535386439317e-08, 3.077806936222375e-08, 3.444018783892709e-08, 3.853804228007881e-08, 4.312347858632976e-08, 4.8254511525794925e-08, 5.399605873472421e-08, 6.042076205299948e-08, 6.760990658596893e-08, 7.565444912054896e-08, 8.465616890708036e-08, 9.472895536658581e-08, 1.0600024901545148e-07, 1.1861265383805813e-07, 1.3272574150704306e-07, 1.4851807027811471e-07, 1.6618944410240523e-07, 1.8596344053856448e-07, 2.0809023944764238e-07, 2.3284978825930876e-07, 2.6055534385622537e-07, 2.915574358883792e-07, 3.2624830166105673e-07, 3.6506684870652016e-07, 4.085042078256366e-07, 4.5710994685634524e-07, 5.114990237853254e-07, 5.723595671733721e-07, 6.40461582332131e-07, 7.166666934024132e-07, 8.019390445904994e-07, 8.973574984844655e-07, 1.004129285783172e-06, 1.1236052791337742e-06, 1.2572970843217753e-06, 1.4068961650507032e-06, 1.5742952432774538e-06, 1.7616122458594442e-06, 1.9712171004857976e-06, 2.2057617199135054e-06, 2.468213553868192e-06, 2.761893133106652e-06, 3.0905160806474777e-06, 3.4582401216940206e-06, 3.869717687017736e-06, 4.330154775337729e-06, 4.845376819421247e-06, 5.421902389240493e-06, 6.067025664675357e-06, 6.788908721203608e-06, 7.596684796173615e-06, 8.500573842180649e-06, 9.512011829523614e-06, 1.0643795433672406e-05, 1.1910243938325615e-05, 1.3327380402450951e-05, 1.4913134383425853e-05, 1.6687568781125115e-05, 1.8673133672977014e-05, 2.0894950351501115e-05, 2.3381129157958297e-05, 2.616312513333413e-05, 2.9276135986337166e-05, 3.275954741348903e-05, 3.665743140547921e-05, 4.1019103844337766e-05, 4.589974844612465e-05, 5.136111494323494e-05, 5.747230033969645e-05, 6.43106231238724e-05, 7.19626014990754e-05, 8.052504800862989e-05, 9.010629440453818e-05, 0.0001008275622567425, 0.00011282449664388024, 0.0001262488823297291, 0.00014127056413834515, 0.00015807959582440375, 0.00017688864462617396, 0.00019793568192343826, 0.00022148699404246175, 0.0002478405513006017, 0.000277329777915584, 0.0003103277704761965, 0.00034725201834633334, 0.00038856968572476344, 0.00043480352218904736, 0.00048653847650358036, 0.0005444290973694139, 0.0006092078147498652, 0.0006816942065469339, 0.0007628053678701707, 0.0008535675140895983, 0.0009551289644740663, 0.001068774670683568, 0.0011959424739294295, 0.0013382412964873741, 0.001497471497721792, 0.001675647652164875, 0.0018750240378380175, 0.002098123157292769, 0.0023477676522184803, 0.0026271160153990606, 0.0029397025518451813, 0.0032894820946886764, 0.003680880041582952, 0.0041188483446680715, 0.0046089281624834486, 0.0051573199664978, 0.005770961989241546, 0.006457618006564497, 0.007225975564636586, 0.008085755894456146, 0.009047836904499748, 0.010124390807612151, 0.011329038123386556, 0.012677020004467484, 0.014185391067042256, 0.015873235165205134, 0.017761906839163368, 0.01987530149207875, 0.02224015771381192, 0.024886395576558126, 0.027847495182478157, 0.03116092025269479, 0.03486859211688692, 0.0390174201003799, 0.04365989501916965, 0.04885475329175736, 0.054667720070112856, 0.06117234079183113, 0.06845091167425013, 0.0765955209231047, 0.08570921382905075, 0.09590729649281649, 0.10731879467366379, 0.1200880862184518, 0.13437672772474157, 0.15036549854880454, 0.1682566880192568, 0.18827665479405745, 0.21067869074175258, 0.23574622558069858, 0.26379641282121963, 0.29518414237992807, 0.3303065306336279, 0.36960794472082353, 0.41358562465807586, 0.4627959744020469, 0.5178616014514738, 0.5794791940538968, 0.648428335679216, 0.7255813682805201, 0.8119144291317323, 0.9085198008797973, 1.0166197310637302, 1.1375818959446489, 1.2729367042944966, 1.4243966600704154, 1.5938780289505816, 1.7835250828556384, 1.9957372291966322, 2.2331993680875106, 2.4989158616006892, 2.7962485448432166, 3.128959259768491, 3.5012574498594473, 3.917853417849659, 4.384017920296717, 4.905648852996557, 5.4893458709396805, 6.14249388689863, 6.873356505067178, 7.691180571870728, 8.606313166720222, 9.630332512872709, 10.77619446467773, 12.058396424560241, 13.493160763612499, 15.098640066424169, 16.895146796901894, 18.905410290800173, 21.154864326425056, 23.67196891184254, 26.4885703598335, 29.64030420624753, 33.16704606946634, 37.11341615523067, 41.52934378981023, 46.47069912396545, 51.999999999999986]



newpks=[0.0523084, 0.0583353, 0.0650565, 0.0725521, 0.0809113, 0.0902337, \
0.10063, 0.112224, 0.125155, 0.139575, 0.155656, 0.17359, 0.193591, \
0.215896, 0.240771, 0.268511, 0.299449, 0.33395, 0.372427, 0.415337, \
0.463191, 0.516558, 0.576075, 0.642448, 0.716469, 0.799019, 0.891079, \
0.993747, 1.10824, 1.23593, 1.37833, 1.53714, 1.71424, 1.91176, \
2.13202, 2.37767, 2.65162, 2.95713, 3.29784, 3.67781, 4.10155, \
4.57412, 5.10114, 5.68888, 6.34433, 7.07531, 7.8905, 8.79962, \
9.81349, 10.9442, 12.2051, 13.6114, 15.1796, 16.9286, 18.879, \
21.0542, 23.48, 26.1853, 29.2023, 32.5669, 36.3192, 40.5038, 45.1705, \
50.375, 56.179, 62.6518, 69.8704, 77.9206, 86.8984, 96.9106, 108.076, \
120.529, 134.416, 149.902, 167.174, 186.435, 207.916, 231.871, \
258.587, 288.38, 321.606, 358.661, 399.984, 446.045, 497.405, \
554.678, 618.479, 689.653, 768.961, 857.394, 955.897, 1065.64, \
1187.9, 1324.11, 1475.66, 1644.34, 1831.96, 2040.57, 2272.25, \
2529.47, 2814.83, 3130.88, 3480.62, 3867.2, 4293.74, 4763.25, \
5279.37, 5844.95, 6463.06, 7136.45, 7866.96, 8656.57, 9505.72, \
10414., 11379.6, 12399., 13466.5, 14574.9, 15712.4, 16864.4, 18013.8, \
19137.5, 20210.1, 21197.8, 22065.3, 22777., 23272.4, 23506.3, \
23497.4, 23215.8, 22525., 21419.3, 20023., 18468.1, 16856.6, 15194.4, \
13572.3, 12252.6, 11273., 10454.5, 9589.43, 8504.41, 7175.08, \
5804.54, 4881.51, 4373., 3755.2, 3045.14, 2530.81, 2126.93, 1714.01, \
1369.76, 1124.6, 921.328, 739.966, 589.292, 468.245, 371.237, \
293.297, 230.641, 180.688, 141.117, 109.852, 85.2539, 65.9723, \
50.9128, 39.1954, 30.0964, 23.0553, 17.6249, 13.4433, 10.2334, \
7.77622, 5.89771, 4.46521, 3.37517, 2.54706, 1.91938, 1.44415, \
1.08517, 0.814297, 0.610269, 0.456804, 0.341526, 0.255056, 0.190288, \
0.141797, 0.105558, 0.0785154, 0.0583443, 0.0433101, 0.0321261, \
0.0238079, 0.0176297, 0.0130445, 0.00964373, 0.00712349, 0.00525817, \
0.00387809, 0.00285801]
allks=list(lowks)+ks_read+newks
allpks=lowpks+pks_read+newpks

pkk=interp1d(allks,allpks)

def pkk_from_nb(k):
    return pkk(k)

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
                                               

                                               

def clg(ell,chi1av,sigmax1,chi2av,sigmax2,tresolution,chiresolution,maxfreq):
    bias=1.9
    t1=time.time()
    print(ell)
    kmin=1e-8
    kmax=52
    Nmax=maxfreq
    
    cns,fns=coeffs(to_transform,Nmax,kmin,kmax,bias)
    answer=np.sum(cns*intIW(ell,fns,chi1av,sigmax1,chi2av,sigmax2,tresolution,chiresolution))/(2*np.pi**2)
    print(answer,"in ",time.time()-t1,"seconds")

    return (answer)

def limbercl(ell,chi1av,sigmax1,chi2av,sigmax2,chiresolution):
    chimin=chi2av-5*sigmax2
    chimax=chi2av+5*sigmax1
    
    
    chis=np.linspace(chimin,chimax,chiresolution)
    
    integrand=[1/chi**2*(Wg(chi,sigmax1,chi1av)*Wg(chi,sigmax2,chi2av))*Pk((ell+1/2)/chi,True,True) for chi in chis]
    return integrate.simps(integrand,chis)#,chis,integrand
    
def coeffs2(function,Nmax,kmin,kmax,bias):
    
    kappamin,kappamax=np.log(kmin),np.log(kmax)
    
    ints=np.linspace(-int(Nmax/2),int(Nmax/2),Nmax-Nmax%2+1)
   
    kappas=kappamin+np.linspace(0,kappamax-kappamin,Nmax-Nmax%2)
    delta=kappas[1]-kappas[0]
    width=(Nmax)*delta
       
    samples=[function(np.exp(kappa))*(np.exp(kappa)/kmin)**bias+function(2*kmin)*(2*kmin/kmin)**bias for kappa in kappas]
    ints01=np.array(range(0,Nmax))
    
    cns=np.zeros(len(ints),dtype=np.complex_)
    fns=np.zeros(len(ints),dtype=np.complex_)

    for i in range(0,len(cns)):
        fns[i]=2*np.pi*(0+1j)*ints[i]/width-bias

        #cns[i]=(kmin**(-fns[i]*0))*1/Nmax*np.sum(samples*np.exp(-(2*np.pi*(0+1j)*ints[i]*ints01)/Nmax)) #*1/(np.sqrt(kmin*kmax))**fns[i]
        cns[i]=1/Nmax*kmin**(-fns[i])*np.sum(samples*np.exp(-(2*np.pi*(0+1j)*ints[i]*ints01)/Nmax)) #*1/(np.sqrt(kmin*kmax))**fns[i]
        
    if Nmax%2==0:
        cns[0]=cns[0]/2
        cns[-1]=cns[-1]/2
    return cns,fns

def logf_transform2(function,x,Nmax,xmin,xmax,bias):
    cns,fns=coeffs2(function,Nmax,xmin,xmax,bias)
   # print(cns/(np.sqrt(xmin*xmax))**(fns),fns)
    return np.sum([cns[i]*((x/1))**(fns[i]) for i in range(0,len(cns))]) 





