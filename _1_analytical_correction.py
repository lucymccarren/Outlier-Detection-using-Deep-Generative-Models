
## Computing Log-Likelihood Analytically for Bias Correction
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# Compute_Likelihood_Analytically
        
def Neg_Reconstruction_Error(lambdaa,target):
    # xi = Pixel value of ith pixel in input sample.
    # For Perfect reconstruction: The input pixel value is equal to the pixel reconstructed by decoder.
                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    if lambdaa == float(1/2):
        xi=float(1/2)
    else:
        xi=(lambdaa/(2*lambdaa)-1)+1/(2*np.arctanh(1-(2*lambdaa)))
    
    # Negative Reconstruction Error for cont. Bernoulli visible distribution
    log_pdf=np.log(xi)+target*np.log(lambdaa)+(1-target)*np.log(1-lambdaa)
    
    return -log_pdf

def Analytical_Correction_For_Intensity_Bias(Pixels): 

    Reconstruction_LL = {}; #Dict={}

    # Correcting Bias in the Reconstruction Error for each Pixel.
    for i in range(len(Pixels)):
      func=Neg_Reconstruction_Error
      
      # Using Nelder-Mead Algorithm to maximize log(Pcb(x,Lambdaa))
      x0 = 0.1
      func_min=optimize.fmin(func, x0, args=(Pixels[i],), callback=None)
      func_min = func_min [0]
      
      # The bias in Reconstruction Error is Eliminated
      Reconstruction_LL[Pixels[i]] = -Neg_Reconstruction_Error(func_min,Pixels[i]) 
    
    Reconstruction_LL=np.float(Reconstruction_LL)
    
    # Creating Dictionary to store unique values.
    Dict = {}
    Pixels=Pixels.astype(np.float32)
    for i in range(len(Pixels)):
        Dict[(Pixels[i] * 1000).round().astype(np.int32)] = Reconstruction_LL[Pixels[i]]
   
    # It will take in each element of the array and return the corresponding value from the Dict dictionary.
    # It will quickly access multiple values in a dictionary.
    Correction=np.vectorize(lambda x: Dict[x])
    
    return Reconstruction_LL,Correction

def Plot_Likelihood(Pixels, Reconstruction_LL):
    plt.figure(figsize=(15,15))
    plt.ylabel("Reconstructed LL")
    plt.xlabel("Pixel Intensity")
    plt.plot(Pixels, Reconstruction_LL)
    plt.show()
    
    return

# def Accuracy(probs):
#     for p in probs:
#        for d in datasets:
#            a=probs[p][d]
#            b=probs[p][ID]
#            dataset_prob=np.zeros_like(a)
#            id_data_prob=np.ones_like(b) 
#            likelihood=np.concatenate([a,b])
#            targets=np.concatenate([dataset_prob,id_data_prob])    
    
#     score=[]
#     for l,t in zip(likelihood,targets):
#         score.append(sklearn.metrics.roc_auc_score(targets[t], likelihood[l]))
    
#     total_score=np.sum(score)
#     return total_score


