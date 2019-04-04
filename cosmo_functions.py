import numpy as np

import camb


from scipy.interpolate import interp1d


def redshift(chi):
    return results.redshift_at_comoving_radial_distance(chi)
def a(chi):
    return 1/(1+redshift(chi))
def comoving_distance(z):
    return results.comoving_radial_distance(z, tol=0.00000001)
def dzdchi(chi):
    return results.h_of_z(redshift(chi))


print("setting up cosmo functions")

print("getting params from CAMB")

params=camb.model.CAMBparams()
params.set_cosmology(H0=0.6727*100, ombh2=0.022264244267999996, omch2=0.12055273725599998, omk=0.0)

results=camb.get_background(params)
H0=results.hubble_parameter(0)
c=2.99792458e5 #km/s
omegam=params.omegab+params.omegac
print("params defined")

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
print("Defining growth function..")

_amin = 0.00005   # minimum scale factor THIS SEEMS WAY TOO LOW CHANGE THIS???
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
                
                return _da_interp(a)/_da_interp(1.0)*5*omegam/2 #hy 0.76? 5omegam/2 = 0.75.....
D_camb = deltakz[0,:,0]/deltakz[0,0,0]
_da_interp = interp1d(atab, D_camb, kind='linear')
       
def D_growth_norm1_z0( a):
        # D(a)
                
               
                
                return _da_interp(a)/_da_interp(1.0) #hy 0.76? 5omegam/2 = 0.75.....
def D_growth_chih_normz0(chi):
    
    #input chi in h^-1Mpc
    #need to convert to Mpc -> Chi = chi/h
    h=H0/100
    return D_growth_norm1_z0( a(chi/h))
def D_growth_chi_normz0(chi):
    
    #input chi in h^-1Mpc
    #need to convert to Mpc -> Chi = chi/h
    return D_growth_norm1_z0( a(chi))

#derivative of groth function (find better way?)
def derivgrowth_h(chi):
    delta=chi*0.0001
    return( D_growth_chih_normz0(chi+delta)-D_growth_chih_normz0(chi-delta))/(2*delta)
    
def secondderivgrowth_h(chi):
    delta=chi*0.0001
    return (D_growth_chih_normz0(chi+delta)+D_growth_chih_normz0(chi-delta)-2*D_growth_chih_normz0(chi))/delta**2


chis=np.linspace(0.01,8000,100)
dgrowths=derivgrowth_h(chis)
interp_dgrowthdchi=interp1d(chis,dgrowths)
def dgrowthdchi(chi):
    return interp_dgrowthdchi(chi)

ddgrowths=secondderivgrowth_h(chis)
interp_seconddgrowthdchi=interp1d(chis,ddgrowths)
def d2growthdchi(chi):
    return interp_seconddgrowthdchi(chi)


print("Growth function defined..")

def get_power_spectra(params):

    Plin=camb.get_matter_power_interpolator(params, nonlinear=False,hubble_units=False, k_hunit=False, kmax=100, zmax=1200)
    Plin=Plin.P
    Pnonlin=camb.get_matter_power_interpolator(params, nonlinear=True,hubble_units=False, k_hunit=False, kmax=100, zmax=1200)
    Pnonlin=Pnonlin.P
    
    return Plin,Pnonlin
print("Getting power spectra from CAMB")


Plin,Pnonlin=get_power_spectra(params)

print("Got power spectra")

def fnl_bias_factor(k,chi):
    deltac= 1.42
    return 3*deltac*omegam*(H0/c)**2*(1/transferfunc2(k))*(1/growth(chi)) 

def fnl_bias_factor_without_k(chi):
    deltac= 1.42
    return 3*deltac*omegam*(H0/c)**2*(1/D_growth(a(chi))) 
    

def factor_wo_growth(k):
    deltac= 1.42
    return 3*deltac*omegam*(H0/c)**2*(1/transferfunc2(k))
print("Getting transfer function...")


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
    k=np.atleast_1d(k)
    ks=trans.transfer_data[0,:,0]*( results.hubble_parameter(0)/100)#trans.transfer_data gives comoving k/h. MULTIPLY by H0/100 to get
                                                                    #physical k.
    ts=transferfunct2
    #plt.loglog(ks,ts)
    answer=np.zeros(len(k))
   # k[k<min(ks)]=min(ks)

    tran=interp1d(ks, ts,bounds_error=False,fill_value=0.)
    
    answer[k<min(ks)]=1
    answer[k>min(ks)]=tran(k[k>min(ks)])

    #answer[k<max(ks)]= tran(k[k<max(ks)])
    
   # if k>max(ks):
    #    print (k, "is greater then",max(ks))
   # if k<min(ks):
    #    print (k, "is less then",min(ks))
    return answer
    

print("Got transfer function...")

print("cosmo_functions set up.\n")

