import numpy as np

import camb

import config

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys

def redshift(chi):
    return results.redshift_at_comoving_radial_distance(chi)
def a(chi):
    return 1/(1+redshift(chi))
def comoving_distance(z):
    return results.comoving_radial_distance(z, tol=0.00001)
def dzdchi(chi):
    return results.h_of_z(redshift(chi))





params=camb.model.CAMBparams()
params.set_cosmology(H0=0.6727*100, ombh2=0.022264244267999996, omch2=0.12055273725599998, omk=0.0)

results=camb.get_background(params)
H0=results.hubble_parameter(0)
c=2.99792458e5 #km/s
omegam=params.omegab+params.omegac

def growth(chi): #growth function 
    sf=a(chi)
    aas=np.linspace(1e-3,sf,200)
    
    
    zs= [1/A-1 for A in aas]
  
    Hs=[results.hubble_parameter(z)for z in zs]
    integrand=[1/(aas[i]*Hs[i]/H0)**3 for i in range(0,len(zs))]
  
    integral=np.trapz(integrand,aas)
    
   # print(Hs[-1])
    return (5*omegam/2/H0)*Hs[-1]*integral


def a2z(a):
        return (1.0/a)-1.0
    
    
def z2a(z):
        return 1/(1+z)
    
_amin = 0.001    # minimum scale factor
_amax = 1.0      # maximum scale factor
_na = 512        # number of points in interpolation arrays
atab = np.linspace(_amin,_amax,_na)
_ks = np.logspace(np.log10(1e-5),np.log10(1.),num=100) 
_zs = a2z(atab)
deltakz = results.get_redshift_evolution(_ks, _zs, ['delta_cdm']) #index: k,z,0
    
def D_growth( a):
        # D(a)
                
                D_camb = deltakz[0,:,0]/deltakz[0,0,0]
                _da_interp = interp1d(atab, D_camb, kind='linear')
                
                return _da_interp(a)/_da_interp(1.0)*0.76 #hy 0.76? 5omegam/2 = 0.75.....

       
def D_growth_norm1_z0( a):
        # D(a)
                
                D_camb = deltakz[0,:,0]/deltakz[0,0,0]
                _da_interp = interp1d(atab, D_camb, kind='linear')
                
                return _da_interp(a)/_da_interp(1.0) #hy 0.76? 5omegam/2 = 0.75.....



def get_power_spectra(params):

    Plin=camb.get_matter_power_interpolator(params, nonlinear=False,hubble_units=False, k_hunit=False, kmax=100, zmax=1200)
    Plin=Plin.P
    Pnonlin=camb.get_matter_power_interpolator(params, nonlinear=True,hubble_units=False, k_hunit=False, kmax=100, zmax=1200)
    Pnonlin=Pnonlin.P
    
    return Plin,Pnonlin

'''
CMB lensing efficiency kernel:
'''
def WCMB(chi):
    chi_CMB=comoving_distance(1100)
    chi_S=chi_CMB

    #want to return CMB lensing efficiency kernel; see eg eq 4 of 1607.01761
    return 3/2*(H0/c)**2*omegam*chi/a(chi)*(1-chi/chi_S)

'''
lensing efficiency kernel (general):
'''
def Wk(chi,chi_S):
    return 3/2*(H0/c)**2*omegam*chi/a(chi)*(1-chi/chi_S)



'''
Galaxy lensing efficiency kernel:
'''
zmax=4
Zs=np.linspace(0.01,zmax,10000)
alpha=1.27
beta=1.02
z0=0.5
normalisation_factor_sources=np.trapz(Zs**alpha*np.exp(-(Zs/z0)**beta),Zs)
    

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
    return np.trapz(dnsource_dz(zs),zs)



def make_redshift_boundaries(z0,zmax):
    zs=np.linspace(z0,zmax,100000)
    boundaries=[z0]
    for z in zs:
    #i could make this quicker by using a root-finder eg brentq rather than sampling at every z. would be nice to make this more 
    #accurate by going to a finer z-space but as it is this would take too long. Come back to this if needed?
        if number_of_sources_between_zs(z0,z)>2.6:
            boundaries.append(z)
            z0=z
    boundaries.append(zmax)
    
    return boundaries
boundaries_sources=make_redshift_boundaries(0.01,4)
#print("redshift bins for clustering are",boundaries_sources)

'''
Galaxy lensing efficiency kernel
'''
def Wkgal_lensing(chi,i): # i is the ith redshift bin, starting at 0!!!
    
    zmin=boundaries_sources[i]
    zmax=boundaries_sources[i+1]
    
    z=redshift(chi)
    if(z>zmax):
        return 0
    
    if(z<zmin):
        zsources=np.linspace(zmin,zmax,100)
    else:
        zsources=np.linspace(z,zmax,100)
    integrand=[dnsource_dz(z)*Wk(chi,comoving_distance(z)) for z in zsources]
    
    return 1/number_of_sources_between_zs(zmin,zmax) *np.trapz(integrand,zsources) #number of sources should be 2.6 by definition no?
   
'''
Galaxy density efficiency kernel
'''
Zs=np.linspace(0.2,1,1000)
normalisation_factor_lenses=np.trapz(comoving_distance(Zs)**2/ results.hubble_parameter(Zs),Zs)

def dnilensdz(z):
    nlens_total=0.25
    
    # is my normalisation factor okay?
    return nlens_total*comoving_distance(z)**2/ results.hubble_parameter(z)/normalisation_factor_lenses


def density_lenses_bini(i):
    if i>0 and i<1+config.lensing_bins:
        
        minzs=config.minzs
        maxzs=config.maxzs
        zmin=minzs[i-1]
        zmax=maxzs[i-1]
        zs=np.linspace(zmin,zmax,100)
        return np.trapz([dnilensdz(z)for z in zs],zs)
    else:
        print("check density_lenses_bini argument; there is no bin", i)
        return 
biases=[0.95/D_growth_norm1_z0(z2a(z))for z in config.minzs]

def bg(z):
    #returns galaxy bias. 
    

    
    
    if z>1 or z<0.2:
         print("galaxy bias is ?? at ",z)
         return
    i=0

    
    while i<config.lensing_bins and z>=config.minzs[i]:
        i=i+1
        
    return biases[i-1]
    
    

density_lenses=[density_lenses_bini(i) for i in range(1,config.lensing_bins+1)]


def Wgali_clustering(chi,i):
    
    boundaries_lenses=config.lens_boundaries

    #i is the i-th redshift bin, starting at 1, such that i=1 corresponds to (0.2,0.4)
    z=redshift(chi)
    if(z<boundaries_lenses[i-1]):
        return 0
    if (z>boundaries_lenses[i]):
        return 0
    ni=density_lenses[i-1]

    return bg(z)*1/ni*dnilensdz(z)*dzdchi(z)



def W(spec,chi):
    
    if spec=="CMB lensing":
        return WCMB(chi)
    
    if spec[0:5]=="shear":
        i=int(spec[6:])
        return Wkgal_lensing(chi,i)
    
    if spec[0:7]=="density":
        i=int(spec[8:])
        return Wgali_clustering(chi,i)
    else:
        print("spec problem")
        
        
upperlimits=[2*np.pi*comoving_distance(minz)/10*H0/100 for minz in config.minzs]
    

def get_Cls(ls,Ws1,Ws2,matter_pk,chis,zs):#specs tells you which power spectrum you want
   
    #ls is the l-values you will compute at
    #Ws1 and Ws2 are the kernels you are integrating
   
    #zs=redshift(chis)
    
    Cls=[]
    
    for l in ls:
        integrand=[Ws1[k]*Ws2[k]*matter_pk(zs[k],(l+1/2)/chis[k])/chis[k]**2 for k in range(0,len(chis))]
        
        Cls.append(np.trapz(integrand,chis))
        
    return np.array(Cls)

#this was really inefficient, recomputing W N^2 times instead of N where N is bin size.
'''
def get_Cls(ls,specs,matter_pk):#specs tells you which power spectrum you want
    if specs==["CMB lensing","CMB lensing"]:
     #   chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(1100)),100)
         chis=np.linspace(0.01,comoving_distance(1100),100)
         zs=[redshift(chi) for chi in chis]
    else:
       # chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(4)),50)
        zs=np.linspace(0.01,4,1000)
        chis=comoving_distance(zs)

    #zs=redshift(chis)

    Ws=[W(specs[0],chi)*W(specs[1],chi)/chi**2 for chi in chis]
    
    Cls=[]
    
    for l in ls:
        print(l)
        integrand=[Ws[k]*matter_pk(zs[k],(l+1/2)/chis[k])for k in range(0,len(chis))]
        
        Cls.append(np.trapz(integrand,chis))
        
    return np.array(Cls)
'''


def get_Cl_fnl_squared_deriv(ls,specs,matter_pk):#specs tells you which power spectrum you want
    
    #chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(4)),50)
    #zs=redshift(chis)
    zs=np.linspace(0.2,1,50)
    chis=comoving_distance(zs)

    Ws=[W(specs[0],chi)*W(specs[1],chi)/chi**2 for chi in chis]
    
    Cls=[]
    
    for l in ls:
        Ks=(l+1/2)/chis

        integrand=[2*Ws[k]*matter_pk(zs[k],Ks[k])/bg(zs[k])*1/Ks[k]**2*(bg(zs[k])-1)*factor(Ks[k],chis[k])for k in range(0,len(chis))]
        
        Cls.append(np.trapz(integrand,chis))
        
    return np.array(Cls)

def get_Cl_fnl_deriv(ls,specs,matter_pk):#specs tells you which power spectrum you want
    
    #chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(4)),50)
    #zs=redshift(chis)
    zs=np.linspace(0.2,1,50)
    chis=comoving_distance(zs)

    Ws=[W(specs[0],chi)*W(specs[1],chi)/chi**2 for chi in chis]
    
    Cls=[]
    
    for l in ls:
        Ks=(l+1/2)/chis
       
        integrand=[Ws[k]*matter_pk(zs[k],Ks[k])/bg(zs[k])*1/Ks[k]**2*(bg(zs[k])-1)*factor(Ks[k],chis[k])for k in range(0,len(chis))]
        
        Cls.append(np.trapz(integrand,chis))
        
    return np.array(Cls)

    
    

def factor(k,chi):
    deltac= 1.42
    return 3*deltac*omegam*(H0/c)**2*(1/transferfunc2(k))*(1/growth(chi)) 


def clustering_noise(i):
    density_insr=density_lenses_bini(i+1)*(np.pi/180)**-2*60**2 #arcmin^-2 -> sr^-1
    noiseval_lenses=1/density_insr
    return noiseval_lenses


def shear_noise(i):
     density_insr=number_of_sources_between_zs(boundaries_sources[i],boundaries_sources[i+1])*(np.pi/180)**-2*60**2 
     noiseval_sources= 0.26 **2/density_insr #see pg 4 of 1607.01761. should be squared?
     return noiseval_sources
    
    
     
kmax=100


pars = camb.CAMBparams()
pars.set_cosmology(H0=0.6727*100, ombh2=0.022264244267999996, omch2=0.12055273725599998, omk=0.0)

pars.set_matter_power(redshifts=[0], kmax=kmax)
results= camb.get_results(pars)

trans = results.get_matter_transfer_data()
transferfunct2 = trans.transfer_data[camb.model.Transfer_Weyl-1,:,0]
transferfunct2=transferfunct2/transferfunct2[0]

kh=trans.transfer_data[0,:,0] #this gives k/h . multiply by h to get physical k



def transfer(k):   #i want to input a PHYSICAL k 


            Tk_camb_matter = trans.transfer_z('delta_cdm', z_index=0)
            Tk_camb_matter = Tk_camb_matter/Tk_camb_matter[0] 
            Tk_camb_matter_k = trans.q#/(H0/100) #trans.q gives physical k. DIVIDE by H0/100 to get k/h.
            #interpolate to required sampling
            
            interpolation = interp1d(Tk_camb_matter_k,Tk_camb_matter,bounds_error=False,fill_value=0.) #interpolating voer COMOVING ks
            
            return interpolation(k)
        
def transferfunc2(k): #i want to input a PHYSICAL k
    ks=trans.transfer_data[0,:,0]*( results.hubble_parameter(0)/100)#trans.transfer_data gives comoving k/h. MULTIPLY by H0/100 to get
                                                                    #physical k.
    ts=transferfunct2
    #plt.loglog(ks,ts)
    tran=interp1d(ks, ts,bounds_error=False,fill_value=0.)
    if k>max(ks):
        print (k, "is greater then",max(ks))
    if k<min(ks):
        print (k, "is less then",min(ks))
    return tran(k)
    
Plin,Pnonlin=get_power_spectra(params)



def shear_power_spectra(ls):
    shears=["shear_"+str(i)for i in range(0,config.source_bins)]
    shear_powerspectra=np.zeros((len(shears),len(shears),len(ls)))
    
    zs=np.linspace(0.01,4,1000)
    chis=power_spectra.comoving_distance(zs)
    
    Ws=np.zeros((len(shears),len(zs)))
    for i in range(0,len(shears)):

        for j in range(0,len(zs)):
            Ws[i,j]=W(shears[i],chis[j])

    for i in range(0,len(shears)):
        Ws1=Ws[i]

        for j in range(0,len(shears)):
            Ws2=Ws[j]
            if j>=i:
            
                shear_powerspectra[i,j]=get_Cls(ls,Ws1,Ws2,Pnonlin,chis,zs)
            else:
                    shear_powerspectra[i,j]=shear_powerspectra[j,i]
            
        
    return shear_powerspectra


def clustering_powerspectra(ls):
    
    
    
    density_strings=["density_"+str(i)for i in range(1,1+config.lensing_bins)]
    clustering_powerspectra=np.zeros((len(density_strings),len(ls)))
   
    for i in range(0,len(density_strings)):
        cut_off_ls=np.array(ls)[np.array(ls)<upperlimits[i]]
        clustering_powerspectra[i,0:len(cut_off_ls)]=get_Cls(cut_off_ls,[density_strings[i],density_strings[i]],Pnonlin)
    return clustering_powerspectra



def cross_powerspectra(ls):
    shears=["shear_"+str(i)for i in range(0,config.source_bins)]
    density_strings=["density_"+str(i)for i in range(1,config.lensing_bins+1)]

    cross_powerspectra=np.zeros((len(density_strings),len(shears),len(ls)))

    for i in range(0,len(density_strings)):
        cut_off_ls=np.array(ls)[np.array(ls)<upperlimits[i]]

        for j in range(0,len(shears)):
            
            cross_powerspectra[i,j,0:len(cut_off_ls)]=get_Cls(cut_off_ls,[density_strings[i],shears[j]],Pnonlin)
            
    return cross_powerspectra


def Cls(ls):
    print("getting shear power spectra")
    
    shears=shear_power_spectra(ls)
    print("got shear power spectra")
    print("getting clustering power spectra")

    clusterings=clustering_powerspectra(ls)
    print("got clustering power spectra")
    print("getting cross power spectra")

    cross=cross_powerspectra(ls)
    
    print("got cross power spectra")

    return (shears,clusterings,cross)

     
