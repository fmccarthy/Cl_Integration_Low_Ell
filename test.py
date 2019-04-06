#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:32:09 2019

@author: fionamccarthy
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

import sys
sys.path.insert(0, '../')

#import Integrating_Cls_Bessel
#import previous_commit
import time

from scipy import special

import camb
#import covariance_matrix
#import Integrating_Cls_Bessel
#import cosmo_functions
#import this_one_works
#import cosmo_functions

def hypfnearone(ell,nu,t):
    #hypergemetric function evaluated near one as needed to compute t near 1. see appendxix B of 1705.05022
    
    ##equation B5 of 1705.05022 as equation b6 did not work for me

    
    ttilde=-(1-t**2)**2/(4*t**2)
    t1=-(1-t**2)**2
    t2=(4*t**2)
    tt=t1/t2
    return ttilde,tt
 
    
    
ells=np.concatenate((np.arange(2,10,dtype=float),np.logspace(np.log10(12),np.log10(5000),100)))
ells=np.trunc(ells)

maxfreq=Nmax=200
kmin=1e-8
kmax=52
bias=1.9
fns=np.ones(200)*(-2+1j)
#cns=cns[0:int(maxfreq/2)+1]
kappamin,kappamax=np.log(kmin),np.log(kmax)

period=kappamax-kappamin


    
ints=np.linspace(-int(Nmax/2),int(Nmax/2),Nmax+1-Nmax%2)
fns=np.zeros(len(ints),dtype=np.complex_)

fns=2*np.pi*(0+1j)*ints/period-bias

fns=fns[0:int((maxfreq)/2)+1]


ts=np.linspace(1e-5,1-1e-5,50)

mints=np.zeros((len(ells[ells<10]),len(fns[0:100])))
#if np.max(ells>10):
 #       mints[ells>10,:]=Integrating_Cls_Bessel.interpolated_minT(ells[ells>10],fns)
    
newts=(1-mints[:,:,np.newaxis])*ts+mints[:,:,np.newaxis]
hights=newts.copy()
hights[newts<0.7]=0.9

ttilde_1=hypfnearone(ells[ells<10][:,np.newaxis,np.newaxis],fns[np.newaxis,0,np.newaxis],hights[0:25])[0]
ttilde_2=hypfnearone(ells[ells<10][:,np.newaxis,np.newaxis],fns[np.newaxis,0,np.newaxis],hights[0:25])[-1]
print(ttilde_1/ttilde_2)