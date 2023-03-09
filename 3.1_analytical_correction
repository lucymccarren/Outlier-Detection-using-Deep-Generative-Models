
## Computing Log-Likelihood Analytically for Bias Correction

import sklearn.metrics
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

class Compute_Likelihood_Analytically():
    
    def __init__(self,lambdaa,target,Pixels,dataset,ID):
        self.lambdaa=lambdaa # Optimal lambda value for Perfect Reconstruction
        self.target=target
        self.Pixels=Pixels
        self.dataset=dataset
        self.ID=ID
    
    def Neg_Reconstruction_Error(self):
        # xi = Pixel value of ith pixel in input sample.
        # For Perfect reconstruction: The input pixel value is equal to the pixel reconstructed by decoder.
                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        if self.lambdaa == float(1/2):
            xi=float(1/2)
        else:
            xi=(self.lambdaa/(2*self.lambdaa)-1)+1/(2*np.arctanh(1-(2*self.lambdaa)))
        
        # Negative Reconstruction Error for cont. Bernoulli visible distribution
        log_pdf=np.log(xi)+self.target*np.log(self.lambdaa)+(1-self.target)*np.log(1-self.lambdaa)
        
        return -log_pdf
    
    def Analytical_Correction_For_Intensity_Bias(self): 

        Reconstruction_LL = {}; #Dict={}

        # Correcting Bias in the Reconstruction Error for each Pixel.
        for i in range(len(self.Pixels)):
          func=self.Neg_Reconstruction_Error
          
          # Using Nelder-Mead Algorithm to maximize log(Pcb(x,Lambdaa))
          x0 = 0.1
          func_min=optimize.fmin(func, x0, args=(self.Pixels[i],), callback=None)
          func_min = func_min [0]
          
          # The bias in Reconstruction Error is Eliminated
          Reconstruction_LL[self.Pixels[i]] = -self.Neg_Reconstruction_Error(func_min,self.Pixels[i]) 
        
        Reconstruction_LL=np.float(Reconstruction_LL)
        
        # Creating Dictionary to store unique values.
        Dict = {}
        Pixels=self.Pixels.astype(np.float32)
        for i in range(len(Pixels)):
            Dict[(Pixels[i] * 1000).round().astype(np.int32)] = Reconstruction_LL[Pixels[i]]
       
        # It will take in each element of the array and return the corresponding value from the Dict dictionary.
        # It will quickly access multiple values in a dictionary.
        Correction=np.vectorize(lambda x: Dict[x])
        
        return Reconstruction_LL,Correction
    
    def Plot_Likelihood(self,Reconstruction_LL):
        plt.figure(figsize=(15,15))
        plt.ylabel("Reconstructed LL")
        plt.xlabel("Pixel Intensity")
        plt.plot(self.Pixels, Reconstruction_LL)
        plt.show()
        
        return
    
    def Accuracy(self,probs):
        for p in probs:
           for d in self.datasets:
               a=probs[p][d]
               b=probs[p][self.ID]
               dataset_prob=np.zeros_like(a)
               id_data_prob=np.ones_like(b) 
               likelihood=np.concatenate([a,b])
               targets=np.concatenate([dataset_prob,id_data_prob])    
        
        score=[]
        for l,t in zip(likelihood,targets):
            score.append(sklearn.metrics.roc_auc_score(targets[t], likelihood[l]))
        
        total_score=np.sum(score)
        return total_score
        
if __name__ == "__main__":
    
    # Creating evenly-spaced Pixels over a specified interval.
    # Taking number of samples= 999, as specified in the paper.
    N=999; m=1e-3
    Pixels = np.linspace(start=m, stop=1-m, num=N, endpoint=True)
    
    CL= Compute_Likelihood_Analytically(lambdaa,target,Pixels)
    LL=CL.Neg_Reconstruction_Error()
    Reconstruction_LL, Correction = CL.Analytical_Correction_For_Intensity_Bias()
    CL.Plot_Likelihood(Reconstruction_LL)
    CL.Accuracy(probs)
                

