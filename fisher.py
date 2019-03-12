import numpy as np

import sys


sys.path.insert(0, '../Code/')

import power_spectra

import config

import matplotlib.pyplot as plt
  
from scipy.interpolate import interp1d
 

class Fisher_forecast(object):
    

    def __init__(self):
        
        self.lensing_bins=config.lensing_bins
        
        self.source_bins=config.source_bins
        
        self.cross=0
        
        self.biases=[0.95/power_spectra.growth(power_spectra.comoving_distance(z))for z in config.minzs]
        
        self.dCldalpha=0  


        
    def setup_cls(self,ls,recompute=False):
        
        self.ls=ls
        
        #self.all_ells=range(ls[0],ls[-1])
        self.all_ells=ls
        
        if recompute:
            
           self.shears,self.clusterings,self.cross=power_spectra.Cls(ls)
           self.covariance_matrix=self.covariance(ls)
       
        else:
             covariancefile=np.load("../saved_arrays/"+str(config.lensing_bins)+"_bins_lmin_"+str(self.ls[0])+".npz")
             
             self.covariance_matrix=covariancefile["cov"]
            
        
             self.dCldalpha=covariancefile["dcldalpha"]
             self.dCldalpha_no_lensing=self.dCldalpha[0:config.lensing_bins,0:config.lensing_bins,:]

        self.interpolate_covariancematrix()
         
        self.covariance_matrix_nolensing=self.covariance_matrix[0:config.lensing_bins,0:config.lensing_bins,:]
        self.covariance_matrix_nolensing_interpolated=self.covariance_matrix_interpolated[0:config.lensing_bins,0:config.lensing_bins,:]

        
        self.clustering_noise_matrix=self.clustering_noise(ls)
        self.clustering_noise_matrix_interpolated=self.clustering_noise(self.all_ells)

        
        self.clustering_noise_matrix_nolensing=self.clustering_noise(ls,False)
        self.clustering_noise_matrix_nolensing_interpolated=self.clustering_noise(self.all_ells,False)

        
        self.shear_noise_matrix=self.lensing_noise(ls)
        self.shear_noise_matrix_interpolated=self.lensing_noise(self.all_ells)

        
        self.noise_matrix=self.clustering_noise_matrix+self.shear_noise_matrix
        self.noise_matrix_interpolated=self.clustering_noise_matrix_interpolated+self.shear_noise_matrix_interpolated

        
        self.noise_matrix_nolensing=self.clustering_noise_matrix_nolensing
        self.noise_matrix_nolensing_interpolated=self.clustering_noise_matrix_nolensing_interpolated

        
        if recompute:
            self._dCldalpha()

        self.interpolate_dcldalpha()
        
        
        self.Full_Covariance_matrix=self.Full_Cov_Matrix(self.covariance_matrix,self.noise_matrix)
        
        self.Full_Covariance_matrix_interpolated=self.Full_Cov_Matrix(self.covariance_matrix_interpolated,self.noise_matrix_interpolated)

        
        
        self.Full_Covariance_matrix_nolensing=self.Full_Cov_Matrix(self.covariance_matrix_nolensing,self.noise_matrix_nolensing)
        
        self.Full_Covariance_matrix_nolensing_interpolated=self.Full_Cov_Matrix(self.covariance_matrix_nolensing_interpolated,self.noise_matrix_nolensing_interpolated)




    def interpolate_covariancematrix(self):
        
        
        self.covariance_matrix_interpolated=np.zeros((self.covariance_matrix.shape[0],self.covariance_matrix.shape[1],len(self.all_ells)))
        for i in range(0,self.covariance_matrix.shape[0]):
            for j in range(0,self.covariance_matrix.shape[1]):
                interp=interp1d(self.ls,self.covariance_matrix[i,j,:])
                self.covariance_matrix_interpolated[i,j,:]=interp(self.all_ells)
        
        
   
        
        
        
        
   

    def covariance(self,ls,lensing=True):

        covariance=np.zeros((self.lensing_bins+self.source_bins,self.lensing_bins+self.source_bins,len(ls)))
        
        

        for i in range(0,covariance.shape[0]):
            for j in range(0,covariance.shape[0]):
                if i<self.lensing_bins:
                    if j==i:
                        covariance[i,j]=self.clusterings[i] #diagonal entries are C_l^gg

                    elif j<config.lensing_bins:
                        covariance[i,j]=np.zeros(len(ls)) #needs to be zero on the off-diagonal
                    else:
                    #    print(cls_gal_gal_lensing_nonlin[i][j-4] )
                        covariance[i,j]=self.cross[i,j-self.lensing_bins] #C_l^{g\kappa}; i think this is good.
                else:
            
                    if j<config.lensing_bins:
                        covariance[i,j]=self.cross[j,i-self.lensing_bins]
                    else:
                        covariance[i,j]=self.shears[i-self.lensing_bins,j-self.lensing_bins]
        if lensing:
            return covariance
        else:
            return covariance[0:self.lensing_bins,0:self.lensing_bins,:]
                        
                        
        return covariance



    def clustering_noise(self,ls,lensing=True):

        noise=np.zeros((self.lensing_bins+self.source_bins,self.lensing_bins+self.source_bins,len(ls)))


    
        for i in range(0,noise.shape[0]):
            if i<self.lensing_bins:
        
                noise[i,i]=power_spectra.clustering_noise(i)
        
        
        if lensing:
            return noise
        else:
            return noise[0:self.lensing_bins,0:self.lensing_bins,:]
        
    def lensing_noise(self,ls,lensing=True):

        noise=np.zeros((self.lensing_bins+self.source_bins,self.lensing_bins+self.source_bins,len(ls)))


    
        for i in range(0,noise.shape[0]):
            if i>=self.lensing_bins:
        
                noise[i,i]= power_spectra.shear_noise(i-self.lensing_bins)
        
        
        if lensing:
            return noise
        else:
            return noise[0:self.lensing_bins,0:self.lensing_bins,:]
        
    def Full_Cov_Matrix(self,covariance, noise):
       
        return covariance+noise
    
     
    
   
    
        
        
    
    def _dCldalpha(self):
    
        print("getting dcldalpha")
        self.dCldalpha=np.zeros((1+len(self.biases),self.covariance_matrix.shape[0],
                            self.covariance_matrix.shape[1],
                            self.covariance_matrix.shape[2]))   #first index is for the derivative (alpha-index)
                                                                    #last index is for l's
                                                                    #should be symmetric on second, third indices
        
        
        self.dCldalpha_no_lensing=np.zeros((1+len(self.biases),self.covariance_matrix_nolensing.shape[0],
                                       self.covariance_matrix_nolensing.shape[1],
                                       self.covariance_matrix_nolensing.shape[2])) 
        
        self.dCldb()
        
        self.dCldfnl()
        
        print("got dcldalpha")

        
        
        
    def interpolate_dcldalpha(self):
        
        self.dCldalpha_interpolated=np.zeros((1+len(self.biases),self.covariance_matrix_interpolated.shape[0],
                            self.covariance_matrix_interpolated.shape[1],
                            self.covariance_matrix_interpolated.shape[2])) 
        
        self.dCldalpha_no_lensing_interpolated=np.zeros((1+len(self.biases),self.covariance_matrix_nolensing_interpolated.shape[0],
                                       self.covariance_matrix_nolensing_interpolated.shape[1],
                                       self.covariance_matrix_nolensing_interpolated.shape[2])) 
        
        self.dCldb_interpolated()
        
        self.dCldfnl_interpolated()
        
        
        
        print("got dcldalpha")
       
        
    def dCldb(self): #gets derivative wrt bias
 
        for i in range(0,self.lensing_bins): # which parameter we are differentiating against; 
                             # dCldalpha[i:] is the derivative wrt the ith member of pi^alpha
                
            self.dCldalpha[i+1,i,i]=2*self.covariance_matrix[i,i]/self.biases[i]  # this takes care of the top-right diagonal block of the
                                                                        # covariance matrix, C_l^gg, which depends on b^2
    
            self.dCldalpha_no_lensing[i+1,i,i]=2*self.covariance_matrix_nolensing[i,i]/self.biases[i]
    
    
    
    
            for j in range(self.lensing_bins,self.dCldalpha.shape[1]):
        
                self.dCldalpha[i+1,i,j]=self.covariance_matrix[i,j]/self.biases[i] # this hopefully takes care of the top-right block
                                                         # of the covariance matrix, C_l&g\kappa, which depends on one
                                                         # factor of b.
                
                self.dCldalpha[i+1,j,i]=self.covariance_matrix[i,j]/self.biases[i] #hopefuly takes care of the bottom-left block. 
        
    def dCldb_interpolated(self): #gets derivative wrt bias
 
        for i in range(0,self.lensing_bins): # which parameter we are differentiating against; 
                             # dCldalpha[i:] is the derivative wrt the ith member of pi^alpha
                
            self.dCldalpha_interpolated[i+1,i,i]=2*self.covariance_matrix_interpolated[i,i]/self.biases[i]  # this takes care of the top-right diagonal block of the
                                                                        # covariance matrix, C_l^gg, which depends on b^2
    
            self.dCldalpha_no_lensing_interpolated[i+1,i,i]=2*self.covariance_matrix_nolensing_interpolated[i,i]/self.biases[i]
    
    
    
    
            for j in range(self.lensing_bins,self.dCldalpha.shape[1]):
        
                self.dCldalpha_interpolated[i+1,i,j]=self.covariance_matrix_interpolated[i,j]/self.biases[i] # this hopefully takes care of the top-right block
                                                         # of the covariance matrix, C_l&g\kappa, which depends on one
                                                         # factor of b.
                
                self.dCldalpha_interpolated[i+1,j,i]=self.covariance_matrix_interpolated[i,j]/self.biases[i] #hopefuly takes care of the bottom-left block. 
        
    
    
    
    def get_Cl_fnl_squared_deriv(self,ls,specs,matter_pk):#specs tells you which power spectrum you want
    
        #chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(4)),50)
        #zs=redshift(chis)
       
        Ws=[power_spectra.W(specs[0],chi)*power_spectra.W(specs[1],chi)/chi**2 for chi in self.chis]
    
        Cls=[]
    
        for l_index,l in enumerate(ls):
            Ks=(l+1/2)/self.chis

            integrand=[2*Ws[k]*matter_pk(self.zs[k],Ks[k])/power_spectra.bg(self.zs[k])*1/Ks[k]**2*(power_spectra.bg(self.zs[k])-1)*self.factors[k,l_index]for k in range(0,len(self.chis))]
        
            Cls.append(np.trapz(integrand,self.chis))
        
        return np.array(Cls)

    def get_Cl_fnl_deriv(self,ls,specs,matter_pk):#specs tells you which power spectrum you want
    
        #chis=np.logspace(np.log10(comoving_distance(0.01)),np.log10(comoving_distance(4)),50)
        #zs=redshift(chis)
        

        Ws=[power_spectra.W(specs[0],chi)*power_spectra.W(specs[1],chi)/chi**2 for chi in self.chis]
    
        Cls=[]
    
        for l_index,l in enumerate(ls):
            Ks=(l+1/2)/self.chis
       
            integrand=[Ws[k]*matter_pk(self.zs[k],Ks[k])/power_spectra.bg(self.zs[k])*1/Ks[k]**2*(power_spectra.bg(self.zs[k])-1)*self.factors[k,l_index]for k in range(0,len(self.chis))]
        
            Cls.append(np.trapz(integrand,self.chis))
        
        return np.array(Cls)

    
    
    def dCldfnl(self):
        
        print("getting dcldfnl")
        
        self.zs=np.linspace(0.2,1,50)
        self.chis=power_spectra.comoving_distance(self.zs)
        self.ks=np.zeros((len(self.zs),len(self.ls)))
        
        
        
        self.factors=np.zeros((len(self.zs),len(self.ls)))
        for i in range(0,len(self.zs)):
            for j in range(0,len(self.ls)):
                k=(self.ls[j]+1/2)/self.chis
                self.factors[i,j]=power_spectra.factor(k[i],self.chis[i])        #i gives z index, j gives l index

        density_strings=["density_"+str(i)for i in range(1,self.lensing_bins+1)]
        
        shearstrings=["shear_"+str(i)for i in range(0,self.source_bins)]

        print("getting dcldfnl^2")

        
        #b^2 derivs:
        for i in range(0,len(density_strings)):
     #       print(i/len(density_strings))
            cut_off_ls=np.array(self.ls)[np.array(self.ls)<power_spectra.upperlimits[i]]

            cls_galgal_deriv=self.get_Cl_fnl_squared_deriv(cut_off_ls,[density_strings[i],density_strings[i]],power_spectra.Pnonlin)
            self.dCldalpha[0,i,i,0:len(cut_off_ls)]=cls_galgal_deriv #putting in Fisher matrix
    
            self.dCldalpha_no_lensing[0,i,i,0:len(cut_off_ls)]=cls_galgal_deriv #putting in Fisher matrix
            
        

        #b^1 derivs:
        print("getting dcldfnl^1")


        for i in range(0,len(density_strings)):
            print(i)
            cut_off_ls=np.array(self.ls)[np.array(self.ls)<power_spectra.upperlimits[i]]

            for j in range(0,config.source_bins):
                print(i,j)
              #  print(i/len(density_strings),j/len(self.shears))
       
                cls_galgal_deriv=self.get_Cl_fnl_deriv(cut_off_ls,[density_strings[i],shearstrings[j]],power_spectra.Pnonlin)
                self.dCldalpha[0,i,j+self.lensing_bins,0:len(cut_off_ls)]=cls_galgal_deriv #putting in Fisher matrix
    
    def dCldfnl_interpolated(self):
        
        
        dcldfnl=self.dCldalpha[0]
        
        dcldfnl_interpolated=np.zeros((dcldfnl.shape[0],dcldfnl.shape[1],len(self.all_ells)))
        
        for i in range(0,dcldfnl.shape[0]):
            for j in range(0,dcldfnl.shape[1]):
                interp=interp1d(self.ls,dcldfnl[i,j,:])
                dcldfnl_interpolated[i,j,:]=interp(self.all_ells)
        self.dCldalpha_interpolated[0]=dcldfnl_interpolated
        
        self.dCldalpha_no_lensing_interpolated[0]=dcldfnl_interpolated[0:config.lensing_bins,0:config.lensing_bins,:]
        
    def size_of_covmatrix(self,ls,ell_index):
    
    
            l=ls[ell_index]
            
            
            upperlimits=[2*np.pi*power_spectra.comoving_distance(minz)/10*power_spectra.H0/100 for minz in config.minzs]
            
            index=0
            
            while(index<self.lensing_bins):
            
                if l<upperlimits[index]:
                    return self.lensing_bins+self.source_bins-index
                else:
                    index=index+1
                    
            return self.source_bins

            
            
    
    def fisher_matrix(self,covariance,lens_noise,clustering_noise,ls,discrete=True): #if discrete is false I will integrate over l instead of sum
    
        Fij_ells=np.zeros((1+self.lensing_bins,1+self.lensing_bins,len(ls)))
    
        for ell_index in range(0,len(ls)):
        
            N=self.size_of_covmatrix(ls,ell_index)  
        
            Cl=covariance[-N: , -N: , ell_index]  # deleting the first row and column if N=13,
                                              # the second row and column if N=12, ec
            Nl=clustering_noise[-N:, -N:, ell_index]+lens_noise[-N: , -N:, ell_index]
                                              # print(noise_lensing[-N: , -N:, ell_index])
            covariance_mat=Cl+Nl
            inverse_covm=np.linalg.inv(covariance_mat)


            for i in range(0,Fij_ells.shape[0]):
        
                for j in range(0,Fij_ells.shape[0]):
        
            
                    first=np.matmul(self.dCldalpha[i,-N:,-N:,ell_index],inverse_covm)
                    second=np.matmul(self.dCldalpha[j,-N:,-N:,ell_index],inverse_covm)
                
                    Fij_ells[i,j,ell_index]=np.trace(np.matmul(first,second))
                
        Fisher=np.zeros((1+self.lensing_bins,1+self.lensing_bins))
                
        for i in range(0,1+self.lensing_bins):
        
            for j in range(0,1+self.lensing_bins):
         
                if discrete:
                    Fisher[i,j]=np.sum([(2*ls[k]+1)*Fij_ells[i,j,k] for k in range(0,len(ls))])
                else:
                    Fisher[i,j]=np.trapz([ls[k]*Fij_ells[i,j,k]/(2*np.pi) for k in range(0,len(ls))],ls)
    
        return(18000/41253 *Fisher/(2))
        
    def fisher_matrix_interpolated(self,covariance,lens_noise,clustering_noise,ls,discrete=True): #if discrete is false I will integrate over l instead of sum
    
        Fij_ells=np.zeros((1+self.lensing_bins,1+self.lensing_bins,len(ls)))
    
        for ell_index in range(0,len(ls)):
        
            N=self.size_of_covmatrix(ls,ell_index)  
        
            Cl=covariance[-N: , -N: , ell_index]  # deleting the first row and column if N=13,
                                              # the second row and column if N=12, ec
            Nl=clustering_noise[-N:, -N:, ell_index]+lens_noise[-N: , -N:, ell_index]
                                              # print(noise_lensing[-N: , -N:, ell_index])
            covariance_mat=Cl+Nl
            inverse_covm=np.linalg.inv(covariance_mat)


            for i in range(0,Fij_ells.shape[0]):
        
                for j in range(0,Fij_ells.shape[0]):
        
            
                    first=np.matmul(self.dCldalpha_interpolated[i,-N:,-N:,ell_index],inverse_covm)
                    second=np.matmul(self.dCldalpha_interpolated[j,-N:,-N:,ell_index],inverse_covm)
                
                    Fij_ells[i,j,ell_index]=np.trace(np.matmul(first,second))
                
        Fisher=np.zeros((1+self.lensing_bins,1+self.lensing_bins))
                
        for i in range(0,1+self.lensing_bins):
        
            for j in range(0,1+self.lensing_bins):
         
                if discrete:
                    Fisher[i,j]=np.sum([(2*ls[k]+1)*Fij_ells[i,j,k] for k in range(0,len(ls))])
                else:
                    Fisher[i,j]=np.trapz([ls[k]*Fij_ells[i,j,k]/(2*np.pi) for k in range(0,len(ls))],ls)
    
        return(18000/41253 *Fisher/(2))


    def fisher_matrix_nolensing_summing(self,covariance,clustering_noise,ls,discrete=True):
    
        Fij_ells=np.zeros((1+self.lensing_bins,1+self.lensing_bins,len(ls)))
    
        for ell_index in range(0,len(ls)):
        
            N=self.size_of_covmatrix(ls,ell_index)
        
            if N==self.source_bins:
                Fij_ells[:,:,ell_index]=0
            
            else:
            
                Cl=np.zeros((N-self.source_bins,N-self.source_bins))
        
                Cl=covariance[-N:self.lensing_bins , -N:self.lensing_bins , ell_index]  # deleting the first row and column if N=13,
                                                    # the second row and column if N=12, ec
            
                Nl=clustering_noise[-N:self.lensing_bins , -N:self.lensing_bins, ell_index]
        
                covariance_mat=Cl+Nl
                inverse_covm=np.linalg.inv(covariance_mat)

                for i in range(0,Fij_ells.shape[0]):
        
                    for j in range(0,Fij_ells.shape[0]):
        
            
                        first=np.matmul(self.dCldalpha[i,-N:self.lensing_bins,-N:self.lensing_bins,ell_index],inverse_covm)
                        second=np.matmul(self.dCldalpha[j,-N:self.lensing_bins,-N:self.lensing_bins,ell_index],inverse_covm)
                
                        Fij_ells[i,j,ell_index]=np.trace(np.matmul(first,second))
        Fisher=np.zeros((1+self.lensing_bins,1+self.lensing_bins))
    
        for i in range(0,1+self.lensing_bins):
        
            for j in range(0,1+self.lensing_bins):
          
                if discrete:
                    Fisher[i,j]=np.sum([(2*ls[k]+1)*Fij_ells[i,j,k] for k in range(0,len(ls))])
                else:
                    Fisher[i,j]=np.trapz([ls[k]*Fij_ells[i,j,k]/(2*np.pi) for k in range(0,len(ls))],ls)


           
        return(Fisher*18000/41253*1/(2))
        
    def fisher_matrix_nolensing_summing_inter(self,covariance,clustering_noise,ls,discrete=True):
    
        Fij_ells=np.zeros((1+self.lensing_bins,1+self.lensing_bins,len(ls)))
    
        for ell_index in range(0,len(ls)):
        
            N=self.size_of_covmatrix(ls,ell_index)
        
            if N==self.source_bins:
                Fij_ells[:,:,ell_index]=0
            
            else:
            
                Cl=np.zeros((N-self.source_bins,N-self.source_bins))
        
                Cl=covariance[-N:self.lensing_bins , -N:self.lensing_bins , ell_index]  # deleting the first row and column if N=13,
                                                    # the second row and column if N=12, ec
            
                Nl=clustering_noise[-N:self.lensing_bins , -N:self.lensing_bins, ell_index]
        
                covariance_mat=Cl+Nl
                inverse_covm=np.linalg.inv(covariance_mat)

                for i in range(0,Fij_ells.shape[0]):
        
                    for j in range(0,Fij_ells.shape[0]):
        
            
                        first=np.matmul(self.dCldalpha_interpolated[i,-N:self.lensing_bins,-N:self.lensing_bins,ell_index],inverse_covm)
                        second=np.matmul(self.dCldalpha_interpolated[j,-N:self.lensing_bins,-N:self.lensing_bins,ell_index],inverse_covm)
                
                        Fij_ells[i,j,ell_index]=np.trace(np.matmul(first,second))
        Fisher=np.zeros((1+self.lensing_bins,1+self.lensing_bins))
    
        for i in range(0,1+self.lensing_bins):
        
            for j in range(0,1+self.lensing_bins):
          
                if discrete:
                    Fisher[i,j]=np.sum([(2*ls[k]+1)*Fij_ells[i,j,k] for k in range(0,len(ls))])
                else:
                    Fisher[i,j]=np.trapz([ls[k]*Fij_ells[i,j,k]/(2*np.pi) for k in range(0,len(ls))],ls)


           
        return(Fisher*18000/41253*1/(2))
        
        
    def vary_noise(self,noise_factors):

        sigmas=np.zeros((len(noise_factors),2))


        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix*noise_factor,self.clustering_noise_matrix*noise_factor)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix*noise_factor)
        
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]
        
        plt.figure(figsize=(15,4))
        plt.suptitle("Reducing ALL noise; lmin="+str(self.ls[0])+"; number of clustering bins="+str(config.lensing_bins), fontsize=14)

        plt.subplot(121)    
        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.axvline(x=1)
        plt.axhline(y=1)
        # plt.ylim(1e-2,10)

        plt.legend()
        plt.subplot(122)    
        plt.plot(noise_factors,sigmas[:,1]/sigmas[:,0],'o')
        plt.xscale('log')
        plt.axvline(x=1)

        plt.show()



        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix,self.clustering_noise_matrix*noise_factor)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix*noise_factor)
    
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]

        plt.figure(figsize=(15,4))
        plt.suptitle("Reducing ONLY CLUSTERING noise", fontsize=14)

        plt.subplot(121)    
        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.legend()
        plt.axvline(x=1)
        plt.axhline(y=1)
   # plt.ylim(1e-2,10)

        plt.subplot(122) 
        plt.axvline(x=1)

        plt.plot(noise_factors,sigmas[:,1]/sigmas[:,0],'o')
        plt.xscale('log')
        plt.show()



        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix*noise_factor,self.clustering_noise_matrix)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix)
    
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]

        plt.figure(figsize=(15,4))
        plt.suptitle("Reducing ONLY SHEAR noise", fontsize=14)

        plt.subplot(121)    
        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.legend()
        plt.axvline(x=1)
        plt.axhline(y=1)
        # plt.ylim(1e-2,10)

        plt.subplot(122)    
        plt.plot(noise_factors,sigmas[:,1]/sigmas[:,0],'o')
        plt.xscale('log')
        plt.axvline(x=1)

        plt.show()
        
    def vary_noise2(self,noise_factors):

        sigmas=np.zeros((len(noise_factors),2))


        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix*noise_factor,self.clustering_noise_matrix*noise_factor)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix*noise_factor)
        
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]
        
        plt.figure(figsize=(20,4))

        plt.subplot(131)   
        plt.title("Reducing ALL noise; lmin="+str(self.ls[0])+"; number of clustering bins="+str(config.lensing_bins), fontsize=14)

        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.axvline(x=1)
        plt.axhline(y=1)
        plt.xlabel("noise multiplication by factor")
        plt.ylabel("$\sqrt{F_{f_{NL}f_{NL}}^{-1}}$")
        plt.ylim(1e-2,200)

        # plt.ylim(1e-2,10)
        plt.legend()




        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix,self.clustering_noise_matrix*noise_factor)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix*noise_factor)
    
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]


        plt.subplot(132)    
        plt.title("Reducing ONLY CLUSTERING noise", fontsize=14)

        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.legend()
        plt.xlabel("noise multiplication by factor")
        plt.ylabel("$\sqrt{F_{f_{NL}f_{NL}}^{-1}}$")

        plt.axvline(x=1)
        plt.axhline(y=1)
        plt.ylim(1e-2,100)

       

        for i,noise_factor in enumerate(noise_factors):
            fishermatrix=self.fisher_matrix(self.covariance_matrix,self.shear_noise_matrix*noise_factor,self.clustering_noise_matrix)
            fishermatrix_nolensing=self.fisher_matrix_nolensing_summing(self.covariance_matrix,self.clustering_noise_matrix)
    
            sigmas[i,:]=[np.sqrt(np.linalg.inv(fishermatrix)[0,0]),np.sqrt(np.linalg.inv(fishermatrix_nolensing)[0,0])]


        plt.subplot(133) 
        plt.title("Reducing ONLY SHEAR noise", fontsize=14)

        plt.loglog(noise_factors,sigmas[:,0],'o',label="including lensing")
        plt.loglog(noise_factors,sigmas[:,1],'o',label="only clustering")
        plt.legend()
        plt.axvline(x=1)
        plt.axhline(y=1)
        plt.xlabel("noise multiplication by factor")
        plt.ylabel("$\sqrt{F_{f_{NL}f_{NL}}^{-1}}$")
        plt.ylim(1e-2,100)

        plt.savefig("/Users/fionamccarthy/Documents/PI/Projects/FC/TeX_Files/Images/noisevariation_lmin"+str(self.ls[0])+"_clusteringbins_"+str(config.lensing_bins)+".pdf")
        plt.show()
        
        
    
    
        
        
    def optimal_correlation_coefficient(self,covariance,lensing_noise,clustering_noise):
    
    # we want this to return the optimal corellation coefficients for a linear combination
    # of clutering fields with each shear field.
    # see appendix of 1502.05356
    
    
        optimal_corr_coeffs=np.zeros((config.source_bins,len(self.ls)))
    
        for i in range(0,config.source_bins):
        
            ks=self.covariance_matrix[i+config.lensing_bins,:config.lensing_bins,:] #cross corelation of all the clustering bins with one shear bin i
    
            Qs=self.covariance_matrix[:config.lensing_bins,:config.lensing_bins,:]+lensing_noise[:config.lensing_bins:,:config.lensing_bins,:] +clustering_noise[:config.lensing_bins:,:config.lensing_bins,:]
                     # all the clustering power spectra    
                     
            Qinverse=np.zeros(Qs.shape)
    
    
            for l in range(0,len(self.ls)):
                Qinverse[:,:,l]=np.linalg.inv(Qs[:,:,l])
        
            
                optimal_corr_coeffs[i,l]=np.matmul(ks[:,l],np.matmul(Qinverse[:,:,l],ks[:,l]))/(covariance[i+config.lensing_bins,i+config.lensing_bins,l]+lensing_noise[i+config.lensing_bins,i+config.lensing_bins,l]+clustering_noise[i+config.lensing_bins,i+config.lensing_bins,l])
            
        return optimal_corr_coeffs
    
    
    

Fisher = Fisher_forecast()

   