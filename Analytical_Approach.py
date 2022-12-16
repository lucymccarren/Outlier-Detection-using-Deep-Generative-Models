import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def Outlier_Detection_Analytically(log_probs,log_probs1):
  
  metrics = defaultdict(lambda: defaultdict(dict))
 
  for id_data in log_probs:
    for ood_data in log_probs[id_data]:
      
      test_list1=log_probs[id_data][ood_data]['orig_probs']
      test_list2=log_probs[id_data][id_data]['orig_probs']
      
      test_list3=log_probs[id_data][ood_data]['corr_probs']
      test_list4=log_probs[id_data][id_data]['corr_probs']

      Labels = np.concatenate([np.zeros_like(test_list1),np.ones_like(test_list2)]) 
      Log_Probs_Original = np.concatenate([test_list1,test_list2])
      Log_Probs_Corrected = np.concatenate([test_list3,test_list4])
      
      # ROC CURVE
      FPR, TPR, T = roc_curve(Labels, Log_Probs_Original, pos_label=1, drop_intermediate=False)
      FPR1, TPR1, T1 = roc_curve(Labels, Log_Probs_Corrected, pos_label=1, drop_intermediate=False)
      
      # ROC SCORE
      Original_ROC = roc_auc_score(Labels, Log_Probs_Original)
      Corrected_ROC = roc_auc_score(Labels, Log_Probs_Corrected)
      
      # PRECISION SCORE
      Original_PRC = average_precision_score(Labels, Log_Probs_Original)
      Corrected_PRC = average_precision_score(Labels, Log_Probs_Corrected)  
      
      # COMPUTING ORIIGNAL AND CORRECTED FALSE/TRUE POSITIVE RATE
      index_original = np.argmax(TPR>0.8)        
      x = np.array((TPR[index_original-1], TPR[index_original]))
      y = np.array((FPR[index_original-1], FPR[index_original]))    
      f = interp1d(x,y)
      FPR_original = f(0.8)
      
      index_corrected = np.argmax(TPR1>0.8)  
      x1 = np.array((TPR1[index_corrected-1], TPR1[index_corrected]))
      y1 = np.array((FPR1[index_corrected-1], FPR1[index_corrected]))    
      f1 = interp1d(x1,y1)
      FPR_corrected = f1(0.8)
      
      metrics[id_data][ood_data]['orig_roc'] = Original_ROC*100
      metrics[id_data][ood_data]['orig_prc'] = Original_PRC*100
      metrics[id_data][ood_data]['orig_fpr'] = FPR_original*100
      
      metrics[id_data][ood_data]['corr_roc'] = Corrected_ROC*100
      metrics[id_data][ood_data]['corr_prc'] = Corrected_PRC*100
      metrics[id_data][ood_data]['corr_fpr'] = FPR_corrected*100
      

  for id_data in log_probs1:
    for ood_data in log_probs1[id_data]:   
        test_listt1=log_probs[id_data][ood_data]['orig_probs']
        test_listt2=log_probs[id_data][id_data]['orig_probs']
        
        test_listt3=log_probs[id_data][ood_data]['corr_probs']
        test_listt4=log_probs[id_data][id_data]['corr_probs']
        
        Log_Probs_Original1 = np.concatenate([test_listt1,test_listt2])
        Log_Probs_Corrected1 = np.concatenate([test_listt3,test_listt4])
      
  print("...",metrics[id_data][ood_data]['orig_roc'],metrics[id_data][ood_data]['corr_roc'])
  print("...",metrics[id_data][ood_data]['orig_prc'],metrics[id_data][ood_data]['orig_prc'])
  print("...",metrics[id_data][ood_data]['orig_fpr'],metrics[id_data][ood_data]['corr_fpr'])
         
  print('Original AUC: %.3f' % Original_ROC)
  print("Original Precision",Original_PRC)
  print('Correct AUC: %.3f' % Corrected_ROC)
  print("Corrected Precision",Corrected_PRC)
  
  # Creating ROC curve
  plt.plot(fpr,tpr,color='blue',label='Without Bias Correction',linestyle='dashdot')
  plt.plot(fpr1,tpr1,color='orange',label='With Bias Correction')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.title('AUROC Curves')
  plt.legend()
  plt.show()
  
  sns.kdeplot(Log_Probs_Original,shade=True,color="pink",label="celeb_a(OOD)")
  sns.kdeplot(Log_Probs_Original1,shade=True,color="blue",label="svhn_cropped (ID)")
  plt.xlabel("Log-Likelihood (LL)")
  plt.legend(loc='upper right')
  plt.show()
  
  sns.kdeplot(Log_Probs_Corrected,shade=True,color="pink",label="celeb_a(OOD)")
  sns.kdeplot(Log_Probs_Corrected1,shade=True,color="blue",label="svhn_cropped(ID)")
  plt.xlabel("With Bias Correction (BC-LL)")
  plt.legend(loc='upper left')
  plt.show()
  
  return metrics,fpr, tpr,fpr1,tpr1

if __name__ == "__main__":
    
    VAE_datasets_colored = ['svhn_cropped'] 
    VAE_datasets_grayscale = ['fashion_mnist']  
   
    log_probs = defaultdict(lambda: defaultdict(dict))
    log_probs1 = defaultdict(lambda: defaultdict(dict))

    for id_data in VAE_datasets_colored:
      for norm in ['pctile-5']:
        with open((f'C:/Users/Shahnawaz/Downloads/probs (1).pkl'),'rb') as f:
          d = pickle.load(f)
          print(d['orig_probs']['svhn_cropped'])
          
        for i, ood_data in enumerate(VAE_datasets_colored +['celeb_a']): # Original Image from ID/ OOD from noisy
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]
        
        for i, ood_data in enumerate(VAE_datasets_colored):
          log_probs1[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs1[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]
    
    cb_grayscale_metrics,fpr, tpr,fpr1,tpr1 = Outlier_Detection(log_probs,log_probs1)

 
        
