import os
import sys
import pickle
from absl import app
import tensorflow as tf
from collections import defaultdict
# sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/dataset_prep.py')
import dataset_prep

# sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/CVAE_Network.py')
import CVAE_Network

# sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/Supporting_Function.py')
import Supporting_Function
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Dataset used to train the model
# dataset = ['mnist','grayscale']
# dataset = ['fashion_mnist','grayscale']
# dataset = ['emnist/letters','grayscale']
# dataset = ['sign_lang','grayscale']
# dataset = ['noise','grayscale']

# dataset = ['cifar10','color']
dataset = ['svhn_cropped','color']
# dataset = ['gtsrb','color']
# dataset = ['celeb_a','color']
# dataset = ['compcars','color']
# dataset = ['kitti','color']
  
# datasets used to evaluate the model
datasets_eval = ['cifar10','svhn_cropped','celeb_a','kitti','noise']
# datasets_eval = ['mnist','fashion_mnist','emnist/letters','noise']
  
def Model_Training(argv):

  train = False
  eval = True

  ##### Normalization 
  Normalization_Type = 'contrast_norm'
  # Normalization_Type = 'same' 
  # Normalization_Type = tf.keras.layers.BatchNormalization
  
  ##### Choosing Dataset Type
  for i in range(len(dataset)-1):
    if dataset[1] == 'color':
      mode = dataset[1]
      Channels=3
    elif dataset[1] == 'grayscale':
      mode = dataset[1]
      Channels=1
    else:
        print("ERROR: Choose Suitable Dataset Type")
  
  for i in range(len(dataset)-1):
      dataset_name = dataset[0]
          
  ##### Giving Some Specifications as an Input
  
  print("----------SPECIFICATIONS----------")
  Batch_Size = 64
  Number_of_Filters = 32
  Latent_Dimension = 100
  Visible_Distribution = 'categorical'
  # Visible_Distribution = 'cont_bernoulli'
  
  print("Name of the Dataset:",str(dataset_name))
  print('Datasets used to evaluate the model:', datasets_eval)
  print("Batch Size:",Batch_Size)
  print("Distribution:",Visible_Distribution)
  print("Number of Filters:",Number_of_Filters)
  print("Number of Channels:",Channels)
  print('Normalization Type:', Normalization_Type)
  print("Latent Dimension:",Latent_Dimension)
  print(" ")
                                        
  ##### Fetching Training, Validation & Testing Dataset                                    
  Training, Validation, Testing = dataset_prep.get_dataset(str(dataset_name), Batch_Size,mode,normalize=None,dequantize=False,shuffle_train=True,visible_dist=Visible_Distribution)
  
  ##### Fetching Variational Autoencoder Model
  Variational_Autoencoder = CVAE_Network.CVAE(input_shape=(Number_of_Filters, Number_of_Filters, Channels),num_filters=Number_of_Filters,latent_dim=Latent_Dimension,visible_dist=Visible_Distribution)
 
  ##### Optimizing & Compiling to imrpove the Model's Accuracy and reduce the Loss 
  Directory_Model = os.path.join('.\\test' , f'Models\\{mode}\\{dataset_name}\\{Normalization_Type}_{Visible_Distribution}_nf_{Number_of_Filters}_zdim_{Latent_Dimension}')
  os.makedirs(Directory_Model, exist_ok=True)
  checkpoint_path = os.path.join(Directory_Model, 'weights_{epoch:02d}.hdf5')
  Optimizer = tf.optimizers.Adam(learning_rate=5e-4)
  Variational_Autoencoder.compile(optimizer=Optimizer,loss={'KL Divergence Loss': Variational_Autoencoder.kl_divergence_loss,'Decoder Loss': Variational_Autoencoder.decoder_nll_loss})
  
  ##### Initializing the Training & Evaluation
  
  if train:
    Epochs=50 
    print("----------Initializing the Training----------")
    # Creating a callback to save the model's weights only during training
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_best_only=True)
    Variational_Autoencoder.fit(Training,epochs=Epochs,validation_data=Validation,callbacks=[callback])

  if eval:  
    print("----------Initializing the Evaluation----------")
    print('Directory_Model', Directory_Model)

    weights = tf.io.gfile.listdir(Directory_Model)    
    if not weights:
      raise ValueError('Weights not found. Please train the model first.')
    checkpoint_path = os.path.join(Directory_Model, weights[-1])
    print('checkpoint_path', checkpoint_path)

    # Create a folder to save the probabilities
    folder_probs = os.path.join('.\\test' , f'Probs\\{Visible_Distribution}\\{mode}_{dataset_name}_nf_{Number_of_Filters}_zdim_{Latent_Dimension}')
    os.makedirs(folder_probs, exist_ok=True)

    
    Variational_Autoencoder.build([None]+list((Number_of_Filters, Number_of_Filters, Channels)))
      
    Variational_Autoencoder.load_weights(checkpoint_path)
      
    Variational_Autoencoder.compute_corrections(Training)
      
    Variational_Autoencoder.evaluate(Validation)
    probs_res = Supporting_Function.get_probs(datasets_eval, Variational_Autoencoder ,mode,Normalization_Type, n_samples=5,
                                              split='test',training=False,visible_dist=Visible_Distribution)
                        
     # save the dictionary probs_res to folder_probs
    pickle.dump(probs_res, open(folder_probs+'\\probs.pkl', 'wb'))

if __name__ == '__main__':
  app.run(Model_Training)