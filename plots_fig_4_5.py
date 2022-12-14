import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict

def nested_dict(n, type):
  ''' Creates a dictionary with n levels of a certain dtype'''
  if n == 1:
      return defaultdict(type)
  else:
      return defaultdict(lambda: nested_dict(n-1, type))

def get_log_likelihoods(vis_dist, mode,l,ensembles=False):
  """Loads log likelihoods from probs.pkl files.
  
  Args:
    vis_dist: Visible dist of the model
    mode: "grayscale" or "color"
  Returns:
    A nested dictionary containing the log likelihoods
  """
  
  if mode == 'grayscale':
    datasets = [
      'mnist',
      'fashion_mnist',
      'emnist/letters',
      # 'sign_lang',
    ]
    nf = 32
    cs_hist = 'adhisteq'
  else:
    datasets = [
      'svhn_cropped',
      'cifar10',
      'celeb_a',
      # 'gtsrb',
      # 'compcars',
    ]
    nf = 32
    cs_hist = 'histeq'

  # If the WAIC competing approach is implemented
  log_probs = defaultdict(lambda: defaultdict(dict))
  if ensembles:
    for id_data in datasets:
      for norm in [None]:#[None, 'pctile-5', cs_hist]:
        with open(
            (f'vae_ood/models/{vis_dist}_ensembles/'
            f'{id_data.replace("/", "_")}-{norm}-zdim_20-lr_0.0005-bs_64-nf_{nf}/probs/'
            'probs'+str(l)+'.pkl'),
            'rb') as f:
          d = pickle.load(f)
        for i, ood_data in enumerate(datasets + ['noise']):
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]
    
    # To include the data from dataset kitti
    if mode!='grayscale':
      id_data ='kitti'
      for norm in [None]:#[None, 'pctile-5', cs_hist]:
        with open(
            (f'vae_ood/models/{vis_dist}_ensembles/'
            f'{id_data.replace("/", "_")}-{norm}-zdim_20-lr_0.0005-bs_64-nf_{nf}/probs/'
            'probs'+str(l)+'.pkl'),
            'rb') as f:
          d = pickle.load(f)
        for i, ood_data in enumerate(datasets + ['kitti'] +['noise']):
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]
    
  # Same but not for ensambles (the paper approach)
  else:
    for id_data in datasets:
      for norm in [None,'pctile-5']:#[None, 'pctile-5', cs_hist]:
        with open(
            (f'vae_ood/models/{vis_dist}/'
            f'{id_data.replace("/", "_")}-{norm}-zdim_20-lr_0.0005-bs_64-nf_{nf}/'
            'probs.pkl'),
            'rb') as f:
          d = pickle.load(f)
        for i, ood_data in enumerate(datasets + ['noise']):
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]
    if mode!='grayscale':
      id_data = 'kitti'
      for norm in [None,'pctile-5']:#[None, 'pctile-5', cs_hist]:
        with open(
            (f'vae_ood/models/{vis_dist}/'
            f'{id_data.replace("/", "_")}-{norm}-zdim_20-lr_0.0005-bs_64-nf_{nf}/'
            'probs.pkl'),
            'rb') as f:
          d = pickle.load(f)
        for i, ood_data in enumerate(datasets + ['kitti'] + ['noise']):
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['orig_probs'] = d['orig_probs'][ood_data]
          log_probs[f'{id_data}-{norm}'][f'{ood_data}-{norm}']['corr_probs'] = d['corr_probs'][ood_data]

  return log_probs


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
  return metrics


def plots(datasets,xlabels,ylabels,cb_metrics,cat_metrics,ensemble_metrics,metric='roc',mode='grayscale'):
  ''' Given the metrics of every dataset (cb,cat,emsemble), do the plots and save them'''
  plt.rcParams["figure.figsize"] = (12,8)
  if mode=='grayscale':
    ncols = 3
    nrows = 5
  else:
    ncols = 4
    nrows = 5
  fig, axs = plt.subplots(nrows,ncols)
  text_shift = 3
  if metric=='fpr':
    y_max = 120
    y_min = -20
  else:
    y_max = 120
    y_min = 0 
  
  labela_cb,labelb_cb = 'LL(cBern)','BC-LL(cBern)'
  labela_cb_norm,labelb_cb_norm = 'LL(cBern) norm','BC-LL(cBern) norm'
  labela_cat,labelb_cat = 'LL(categ)', 'BC-LL(categ)'
  labela_cat_norm,labelb_cat_norm = 'LL(categ) norm','BC-LL(categ) norm'
  labela_WAIC = 'WAIC'
  
  for j  in range(ncols):
    key1_cb = datasets[j]+'-None'
    keys2_cb = list(cb_metrics[key1_cb].keys()) + ['avg']
    key1_cb_norm = datasets[j]+'-pctile-5'
    keys2_cb_norm = list(cb_metrics[key1_cb_norm].keys()) + ['avg']
    key1_cat = datasets[j]+'-None'
    keys2_cat = list(cat_metrics[key1_cat].keys()) + ['avg']
    key1_cat_norm = datasets[j]+'-pctile-5'
    keys2_cat_norm = list(cat_metrics[key1_cat_norm].keys()) + ['avg']
    key1_WAIC= datasets[j]+'-None'
    keys2_WAIC= list(ensemble_metrics[key1_WAIC].keys()) + ['avg']

    avg_cb,avg_cb_norm,avg_cat,avg_cat_norm,avg_WAIC = [],[],[],[],[]
    for i in range(nrows):
      key2_cb = keys2_cb[i]     
      key2_cb_norm =  keys2_cb_norm[i]   
      key2_cat = keys2_cat[i]      
      key2_cat_norm =  keys2_cat_norm[i]
      key2_WAIC =  keys2_WAIC[i]

      # For every dataset (except the last row that is the avg) 
      if i!=nrows-1:
        # Since kitti in the plot does not follow the same pattern as the rest of the datasets: 
        if i==3 and j==3 and datasets[j]=='kitti':
          key2_cb = keys2_cb[i+1]
          key2_cb_norm =  keys2_cb_norm[i+1]
          key2_cat = keys2_cat[i+1]
          key2_cat_norm =  keys2_cat_norm[i+1] 
          key2_WAIC =  keys2_WAIC[i+1]

        cb = [round(cb_metrics[key1_cb][key2_cb]['orig_'+metric]), round(cb_metrics[key1_cb][key2_cb]['corr_'+metric])]
        cb_norm = [round(cb_metrics[key1_cb_norm][key2_cb_norm]['orig_'+metric]), round(cb_metrics[key1_cb_norm][key2_cb_norm]['corr_'+metric])]
        cat = [round(cat_metrics[key1_cat][key2_cat]['orig_'+metric]), round(cat_metrics[key1_cat][key2_cat]['corr_'+metric])]
        cat_norm = [round(cat_metrics[key1_cat_norm][key2_cat_norm]['orig_'+metric]), round(cat_metrics[key1_cat_norm][key2_cat_norm]['corr_'+metric])]
        WAIC = [round(ensemble_metrics[key1_WAIC][key2_WAIC]['orig_'+metric])] 

        # Dont take into account id data for the avg
        if i==3 and j==3 and datasets[j]=='kitti':
          avg_cb.append(cb)
          avg_cb_norm.append(cb_norm)
          avg_cat.append(cat)
          avg_cat_norm.append(cat_norm)
          avg_WAIC.append(WAIC)

        # Dont take into account id data for the avg
        elif i!=j:
          avg_cb.append(cb)
          avg_cb_norm.append(cb_norm)
          avg_cat.append(cat)
          avg_cat_norm.append(cat_norm)
          avg_WAIC.append(WAIC)

      # For the last row, show the average
      else:
        cb = [round(np.array(avg_cb).mean(axis=0)[0]),round(np.array(avg_cb).mean(axis=0)[1])]
        cb_norm = [round(np.array(avg_cb_norm).mean(axis=0)[0]),round(np.array(avg_cb_norm).mean(axis=0)[1])]
        cat = [round(np.array(avg_cat).mean(axis=0)[0]),round(np.array(avg_cat).mean(axis=0)[1])]
        cat_norm = [round(np.array(avg_cat_norm).mean(axis=0)[0]),round(np.array(avg_cat_norm).mean(axis=0)[1])]
        WAIC = [round(np.array(avg_WAIC).mean(axis=0)[0])] 

      axs[i,j].plot([0],cb[0], marker = 'o',label=labela_cb)
      axs[i,j].plot([1],cb[1], marker = 'v',label=labelb_cb)
      axs[i,j].plot([0,1],cb,color='k',linewidth=.5)
      axs[i,j].text(0, cb[0]+text_shift, "%d" %cb[0], ha="center")
      axs[i,j].text(1, cb[1]+text_shift, "%d" %cb[1], ha="center")
      axs[i,j].set_ylim(y_min,y_max) 
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])

      axs[i,j].plot([2],cb_norm[0], marker = 'o',label=labela_cb_norm)
      axs[i,j].plot([3],cb_norm[1], marker = 'v',label=labelb_cb_norm)
      axs[i,j].plot([2,3],cb_norm,color='k',linewidth=.5)
      axs[i,j].text(2, cb_norm[0]+text_shift, "%d" %cb_norm[0], ha="center")
      axs[i,j].text(3, cb_norm[1]+text_shift, "%d" %cb_norm[1], ha="center")
      axs[i,j].set_ylim(y_min,y_max) 
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])

      axs[i,j].plot([4],cat[0], marker = 'o',label=labela_cat)
      axs[i,j].plot([5],cat[1], marker = 'v',label=labelb_cat)
      axs[i,j].plot([4,5],cat,color='k',linewidth=.5)
      axs[i,j].text(4, cat[0]+text_shift, "%d" %cat[0], ha="center")
      axs[i,j].text(5, cat[1]+text_shift, "%d" %cat[1], ha="center")
      axs[i,j].set_ylim(y_min,y_max) 
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])

      axs[i,j].plot([6],cat_norm[0], marker = 'o',label=labela_cat_norm)
      axs[i,j].plot([7],cat_norm[1], marker = 'v',label=labelb_cat_norm)
      axs[i,j].plot([6,7],cat_norm,color='k',linewidth=.5)
      axs[i,j].text(6, cat_norm[0]+text_shift, "%d" %cat_norm[0], ha="center")
      axs[i,j].text(7, cat_norm[1]+text_shift, "%d" %cat_norm[1], ha="center")
      axs[i,j].set_ylim(y_min,y_max) 
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])

      axs[i,j].plot([8],WAIC[0], marker = 'o',label=labela_WAIC) 
      axs[i,j].plot([8],WAIC,color='k',linewidth=.5)
      axs[i,j].text(8, WAIC[0]+text_shift, "%d" %WAIC[0], ha="center") 
      axs[i,j].set_ylim(y_min,y_max) 
      axs[i,j].set_xticks([])
      axs[i,j].set_yticks([])

      if i==nrows-1:
        axs[i,j].set_xlabel(xlabels[j])
      if j==0:
        axs[i,j].set_ylabel(ylabels[i])

  if mode=='color':
    axs[4,3].set_xlabel('Kitti VAE')
    plt.rcParams["figure.figsize"] = (14,8)
    axs[0,0].legend(loc='upper center', bbox_to_anchor=(2, 1.8),
              ncol=3, fancybox=True, shadow=True)
  else:
    plt.rcParams["figure.figsize"] = (12,8)
    axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.5, 1.8),
              ncol=3, fancybox=True, shadow=True)

  plt.subplots_adjust(wspace=0, hspace=0)
  plt.savefig('./vae_ood/plots/'+mode+'_'+metric)
  return None


def plot_metrics(mode='grayscale',metric='roc'):
  ''' Given the mode and metric, retrieves all the data for those datasets (and the ensembles) and calls plots with that data'''
  # Load all the models
  lls = []
  for i in range(49,55):
    cb_lls = get_log_likelihoods('cont_bernoulli', mode,i,ensembles=True)
    lls.append(cb_lls)

  # Put all the data from all the models in one list 
  ensembles = nested_dict(3, list)
  for cb_lls in lls:
    for key1 in cb_lls.keys():
      for key2 in cb_lls[key1].keys():
        for key3 in cb_lls[key1][key2].keys():
          ensembles[key1][key2][key3].append(cb_lls[key1][key2][key3])

  # Calculate the results for ensembles
  for key1 in ensembles.keys():
    for key2 in ensembles[key1].keys():
      for key3 in ensembles[key1][key2].keys():
        ensembles[key1][key2][key3] = np.mean(np.array(ensembles[key1][key2][key3]),axis=0) - np.var(np.array(ensembles[key1][key2][key3]),axis=0)
        # print(key1,key2,key3)#,np.round(ensembles[key1][key2][key3],1))

  ensemble_metrics = get_metrics(ensembles)

  cb_lls = get_log_likelihoods('cont_bernoulli', mode,-1)
  cb_metrics = get_metrics(cb_lls)
  cat_lls = get_log_likelihoods('cat',mode,-1)
  cat_metrics = get_metrics(cat_lls)

  if mode=='grayscale':
    ylabels = ['MNIST','FMNIST','EMNIST','Noise','Average'] 
    xlabels = ['MNIST VAE','FMNIST VAE','EMNIST VAE'] 
    datasets = ['mnist','fashion_mnist','emnist/letters']
  elif mode=='color':
    ylabels = ['SVHN','CIFAR10','CelebA','Noise','Average'] 
    xlabels = ['SVHN VAE','CIFAR10 VAE','CelebA VAE','kitti'] 
    datasets = ['svhn_cropped','cifar10','celeb_a','kitti']

  plots(datasets,xlabels,ylabels,cb_metrics,cat_metrics,ensemble_metrics,metric=metric,mode=mode)
  return None
 





metrics = ['roc','prc','fpr'] 

for metric in metrics: 
  # -- GREY DATASETS
  plot_metrics(mode='grayscale',metric=metric)   

  # -- COLOR DATASETS
  plot_metrics(mode='color',metric=metric)

