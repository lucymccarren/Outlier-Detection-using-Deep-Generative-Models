import os
import pickle
from collections import defaultdict
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import tensorflow as tf
def nested_dict(n, type):
  ''' Creates a dictionary with n levels of a certain dtype'''
  if n == 1:
      return defaultdict(type)
  else:
      return defaultdict(lambda: nested_dict(n-1, type))



def get_metrics(log_probs):
  """Computes AUROC, AUPRC and FPR@80 metrics using probs.pkl files.
  
  Args:
    log_probs: original and corrected log likelihoods for all ID-OOD
               pairs as returned by get_log_likelihoods()
  Returns:
    A nested dictionary containing the metrics
  """

  metrics = defaultdict(lambda: defaultdict(dict))
  for m,id_data in enumerate(log_probs):
    for ood_data in log_probs[id_data]:
      labels_concat = np.concatenate(
          [np.zeros_like(log_probs[id_data][ood_data]['orig_probs'][:10000]),
           np.ones_like(log_probs[id_data][id_data]['orig_probs'][:10000])]) 
      lls_concat = np.concatenate(
          [log_probs[id_data][ood_data]['orig_probs'][:10000],
           log_probs[id_data][id_data]['orig_probs'][:10000]])
      orig_roc = roc_auc_score(labels_concat, lls_concat)
      orig_prc = average_precision_score(labels_concat, lls_concat)
      fpr, tpr, thresholds = roc_curve(labels_concat, lls_concat, pos_label=1, drop_intermediate=False)
      ind = np.argmax(tpr>0.8)  
      x = np.array((tpr[ind-1], tpr[ind]))
      y = np.array((fpr[ind-1], fpr[ind]))    
      f = interp1d(x,y)
      orig_fpr = f(0.8)
      metrics[id_data][ood_data]['orig_roc'] = orig_roc*100
      metrics[id_data][ood_data]['orig_prc'] = orig_prc*100
      metrics[id_data][ood_data]['orig_fpr'] = orig_fpr*100
      try:
        lls_concat = np.concatenate(
            [log_probs[id_data][ood_data]['corr_probs'][:10000],
            log_probs[id_data][id_data]['corr_probs'][:10000]])
        corr_roc = roc_auc_score(labels_concat, lls_concat)
        corr_prc = average_precision_score(labels_concat, lls_concat)
        fpr, tpr, thresholds = roc_curve(labels_concat, lls_concat, pos_label=1, drop_intermediate=False)
        ind = np.argmax(tpr>0.8)  
        x = np.array((tpr[ind-1], tpr[ind]))
        y = np.array((fpr[ind-1], fpr[ind]))    
        f = interp1d(x,y)
        corr_fpr = f(0.8)
        metrics[id_data][ood_data]['corr_roc'] = corr_roc*100
        metrics[id_data][ood_data]['corr_prc'] = corr_prc*100
        metrics[id_data][ood_data]['corr_fpr'] = corr_fpr*100
      except:
        metrics[id_data][ood_data]['corr_roc'] = np.nan
        metrics[id_data][ood_data]['corr_prc'] = np.nan
        metrics[id_data][ood_data]['corr_fpr'] = np.nan
  return metrics


def load_ll(id_data, ood_data, norm, mode, Visible_Distribution, Number_of_Filters, Latent_Dimension,l=None):
  # Load data for all the datasets, put it in a dict
  log_probs = defaultdict(lambda: defaultdict(dict))
  for id_data in datasets_id:
    path = f'test\\Probs\\{mode}\\{id_data}\\{norm}_{Visible_Distribution}_nf_{Number_of_Filters}_zdim_{Latent_Dimension}'
    prob_files = tf.io.gfile.listdir(path) 
    print('prob_files',prob_files)
    if l is not None:
      try:
        print('l',l,'loading file',os.path.join(path, prob_files[-l-1]))
        with open(os.path.join(path, prob_files[-l-1]), 'rb') as f:
          d = pickle.load(f)
      except:
        with open(os.path.join(path, prob_files[-1]), 'rb') as f:
          d = pickle.load(f)
        print('error loading last',l,'th file. Loading file',os.path.join(path, prob_files[-1]))
        print('weights:',prob_files)
        print('mode,norm,Visible_Distribution,Number_of_Filters,Latent_Dimension',mode,norm,Visible_Distribution,Number_of_Filters,Latent_Dimension)
    else:
      print('loading file',os.path.join(path, prob_files[-1]))
      with open(os.path.join(path, prob_files[-1]), 'rb') as f:
        d = pickle.load(f)
    for i, ood_data in enumerate(datasets_ood):
      log_probs[f'{id_data}'][f'{ood_data}']['orig_probs'] = d['orig_probs'][ood_data]
      log_probs[f'{id_data}'][f'{ood_data}']['corr_probs'] = d['corr_probs'][ood_data] 
  return  log_probs 




#-------------------------------------------- MAIN --------------------------------------------

''' Choose the datasets and parameters'''

metric = 'fpr' # 'roc' or 'prc' or 'fpr'

datasets_color = ['cifar10','svhn_cropped','kitti','noise']
datasets_grayscale = ['mnist','fashion_mnist','emnist/letters','noise']
datasets_chosen = [
            'cifar10',  
            'svhn_cropped',
            'kitti',
          ]
datasets_chosen = [  
                      'mnist',
                      'fashion_mnist',
                      'emnist/letters',
                    ]
datasets_id = datasets_chosen

# if any of the dataset is in datasets_color, then the mode is color
if any(x in datasets_id for x in datasets_color):
  mode = 'color'
  datasets_ood = datasets_color
elif any(x in datasets_id for x in datasets_grayscale):
  mode = 'grayscale'
  datasets_ood = datasets_grayscale

print('mode', mode)
print('datasets_id', datasets_id)
print('datasets_ood', datasets_ood)

Number_of_Filters = 32
Batch_Size = 64
Latent_Dimension = 20
Normalization_Type = [
                      'None',
                      'contrast_norm',
                      ]
Visible_Distribution = [
                        'cont_bernoulli',
                        'categorical',
                        ]

#--------------------------------------------LOAD PROBS--------------------------------------------

'''Load the log_probs for all the datasets, normalization types, visible distributions'''
log_probs_dict = defaultdict(lambda: defaultdict(dict))
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    log_probs_dict[distr][norm] = load_ll(datasets_id, datasets_ood, norm, mode, distr, 
                        Number_of_Filters, Latent_Dimension,l=None)


print('\nshow the info of log_probs') 
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    log_probs = log_probs_dict[distr][norm]
    print('Distr',distr,'Norm',norm,'\n')
    for d_id in log_probs.keys():
      print('*'*30)
      print('d_id',d_id)
      print('*'*30)
      for d_ood in log_probs[d_id].keys():
        for probs_name, values in log_probs[d_id][d_ood].items():
          print('d_ood',d_ood,probs_name, np.mean(values))
        print('-'*30)
      print('\n')


'''Get the metrics for all the datasets, normalization types, visible distributions'''
metrics_dict = defaultdict(lambda: defaultdict(dict))
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    metrics_dict[distr][norm] = get_metrics(log_probs_dict[distr][norm])

print('\nshow the info of metrics')
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    print('Distr',distr,'Norm',norm,'\n')
    metrics = metrics_dict[distr][norm]
    for d_id in metrics.keys():
      print('*'*30)
      print('d_id',d_id)
      print('*'*30)
      for d_ood in metrics[d_id].keys():
        print('d_ood',d_ood,'\norig_',metric,'corr_',metric,': ',np.round(metrics[d_id][d_ood]['orig_'+metric]),np.round(metrics[d_id][d_ood]['corr_'+metric]))
        # for k, v in metrics[d_id][d_ood].items():
        #   print(k, v)
        # print('-'*30)
      print('\n')


'''Add the average of all the oods to metrics_dict as a new key'''
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    for id_data in datasets_id:
      values_orig = np.array([metrics_dict[distr][norm][id_data][ood_data]['orig_'+metric] 
                                for ood_data in metrics_dict[distr][norm][id_data].keys() if ood_data != id_data])
      values_corr = np.array([metrics_dict[distr][norm][id_data][ood_data]['corr_'+metric] 
                                for ood_data in metrics_dict[distr][norm][id_data].keys() if ood_data != id_data])
      metrics_dict[distr][norm][id_data]['average']['orig_'+metric] = np.mean(values_orig)
      metrics_dict[distr][norm][id_data]['average']['corr_'+metric] = np.mean(values_corr)


#--------------------------------------------ENSEMBLES--------------------------------------------

''' Ensembles '''  
# Append the log probs of all the ensembles to a list
ll_ensembles = []
n_ensembles = 5
print('doing ensembles...')
for i in range(n_ensembles):
  ll_ensembles.append(load_ll(datasets_id, datasets_ood, 'None', mode, 'cont_bernoulli', 
                          Number_of_Filters, Latent_Dimension,l=i))


# Put all the probs from all the models in one list 
ensembles = nested_dict(5, list)
for ensemble in ll_ensembles:
  for id_data in datasets_id:
    for ood_data in datasets_ood:
      for probs_type in ['orig_probs','corr_probs']:
        ensembles['cont_bernoulli']['None'][id_data][ood_data][probs_type].append(ensemble[id_data][ood_data][probs_type])

# Calculate the results for ensembles as the mean of the models - the variance of the models
for id_data in datasets_id:
  for ood_data in datasets_ood:
    for probs_type in ['orig_probs','corr_probs']:
      values = np.array(ensembles['cont_bernoulli']['None'][id_data][ood_data][probs_type])
      ensembles['cont_bernoulli']['None'][id_data][ood_data][probs_type] = np.mean(values,axis=0)-np.var(values,axis=0)


# Calculate the metrics for the ensembles
ensemble_metrics = defaultdict(lambda: defaultdict(dict))
ensemble_metrics = get_metrics(ensembles['cont_bernoulli']['None'])
ensemble_metrics['cont_bernoulli']['None'] = get_metrics(ensembles['cont_bernoulli']['None'])

'''Add the average of all the oods to metrics_dict as a new key'''
for id_data in datasets_id:
  values_orig = np.array([ensemble_metrics['cont_bernoulli']['None'][id_data][ood_data]['orig_'+metric] 
                            for ood_data in ensemble_metrics['cont_bernoulli']['None'][id_data].keys() if ood_data != id_data])
  values_corr = np.array([ensemble_metrics['cont_bernoulli']['None'][id_data][ood_data]['corr_'+metric] 
                            for ood_data in ensemble_metrics['cont_bernoulli']['None'][id_data].keys() if ood_data != id_data])
  ensemble_metrics['cont_bernoulli']['None'][id_data]['average']['orig_'+metric] = np.mean(values_orig)
  ensemble_metrics['cont_bernoulli']['None'][id_data]['average']['corr_'+metric] = np.mean(values_corr)

#--------------------------------------------PLOTS--------------------------------------------

''' Plot the results'''

if mode=='grayscale':
  ylabels = ['MNIST','FMNIST','EMNIST','Noise','Average'] 
  xlabels = ['MNIST VAE','FMNIST VAE','EMNIST VAE'] 
elif mode=='color':
  ylabels = ['CIFAR10','SVHN','Kitti','Noise','Average'] 
  xlabels = ['CIFAR10 VAE','SVHN VAE','Kitti VAE'] 

# Set the parameters for the plots
plt.rcParams["figure.figsize"] = (12,8)
text_shift = 3
# metric = 'roc'
ncols = 3
nrows = 5
fig, axs = plt.subplots(nrows,ncols)
if metric=='fpr':
  y_max = 120
  y_min = -20
else:
  y_max = 120
  y_min = 0 

# Get the labels
label = defaultdict(dict)
label['categorical'] = '(cat)'
label['cont_bernoulli'] = '(cBern)'
label['None'] = ' '
label['contrast_norm'] = 'norm'
label['orig'] = 'LL'
label['corr'] = 'BC-LL'

labels = defaultdict(lambda: defaultdict(dict))
for distr in Visible_Distribution:
  for norm in Normalization_Type:
    for prob_type in ['orig','corr']:
      labels[distr][norm][prob_type] =  label[prob_type] + label[distr] + label[norm] 


for j , id_data in enumerate(datasets_id): 
  for i, ood_data in enumerate(datasets_ood+['average']):
    # Create the pairs of values to plot
    pairs = []
    labels_pairs = []
    for distr in Visible_Distribution:
      for norm in Normalization_Type:
        pairs.append((np.round(metrics_dict[distr][norm][id_data][ood_data]['orig_'+metric]),np.round(metrics_dict[distr][norm][id_data][ood_data]['corr_'+metric])))
        labels_pairs.append((label['orig']+label[distr]+label[norm], label['corr']+label[distr]+label[norm]))
        print('id_data',id_data,'ood_data',ood_data,'distr',distr,'norm',norm,pairs[-1])
      

    # Plot the pairs
    print('len(pairs)',len(pairs),'len(labels_pairs)',len(labels_pairs))
    for k in range(0, len(pairs)*2, 2):
      print('xaxis: (k,k+1): (',k,',',k+1,')')
      print('k//2',k//2,'pairs[k//2]',pairs[k//2])
      axs[i,j].plot([k],pairs[k//2][0], marker = 'o',label=labels_pairs[k//2][0])
      axs[i,j].plot([k+1],pairs[k//2][1], marker = 'v',label=labels_pairs[k//2][1])
      axs[i,j].plot([k,k+1],pairs[k//2],color='k',linewidth=.5)

      axs[i,j].text(k, pairs[k//2][0]+text_shift, "%d" %pairs[k//2][0], ha="center")
      axs[i,j].text(k+1, pairs[k//2][1]+text_shift, "%d" %pairs[k//2][1], ha="center")
    # plot the ensemble
    # ERRROR CON LA PARTE THE OOD = AVERAGE
    # print('ensemble_metrics',id_data,ood_data,ensemble_metrics['cont_bernoulli']['None'][id_data][ood_data])
    WAIC_value = ensemble_metrics['cont_bernoulli']['None'][id_data][ood_data]['orig_'+metric]
    axs[i,j].plot([k+2],WAIC_value, marker = 'o',label='WAIC')
    axs[i,j].text(k+2, WAIC_value+text_shift, "%d" %WAIC_value, ha="center")

    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])      
    axs[i,j].set_ylim(y_min,y_max)

    if i==nrows-1:
      axs[i,j].set_xlabel(xlabels[j])
    if j==0:
      axs[i,j].set_ylabel(ylabels[i])

    print('-'*30)

axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.8, 1.8),
          fancybox=True, shadow=True,ncol=5)


plt.show()
















# ##########################################################################################
# print('\nshow the info of log_probs') 
# for distr in Visible_Distribution:
#   for norm in Normalization_Type:
#     log_probs = log_probs_dict[distr][norm]
#     print('Distr',distr,'Norm',norm,'\n')
#     for d_id in log_probs.keys():
#       print('*'*30)
#       print('d_id',d_id)
#       print('*'*30)
#       for d_ood in log_probs[d_id].keys():
#         for probs_name, values in log_probs[d_id][d_ood].items():
#           print('d_ood',d_ood,probs_name, np.mean(values))
#         print('-'*30)
#       print('\n')

# print('\nshow the info of ensembles') 
# for d_id in ensembles['cont_bernoulli']['None'].keys():
#   print('*'*30)
#   print('d_id',d_id)
#   print('*'*30)
#   for d_ood in ensembles['cont_bernoulli']['None'][d_id].keys():
#     for probs_name, values in ensembles['cont_bernoulli']['None'][d_id][d_ood].items():
#       print('d_ood',d_ood,probs_name, np.mean(values))
#     print('-'*30)
#   print('\n')


# print('\nshow the info of metrics')
# for distr in Visible_Distribution:
#   for norm in Normalization_Type:
#     print('Distr',distr,'Norm',norm,'\n')
#     metrics = metrics_dict[distr][norm]
#     for d_id in metrics.keys():
#       print('*'*30)
#       print('d_id',d_id)
#       print('*'*30)
#       for d_ood in metrics[d_id].keys():
#         print('d_ood',d_ood,'\norig_roc,corr_roc',np.round(metrics[d_id][d_ood]['orig_roc']),np.round(metrics[d_id][d_ood]['corr_roc']))
#         # for k, v in metrics[d_id][d_ood].items():
#         #   print(k, v)
#         # print('-'*30)
#       print('\n')
# ##########################################################################################
