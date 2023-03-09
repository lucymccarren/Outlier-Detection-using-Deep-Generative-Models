import os
import sys
import pickle
from absl import app
import tensorflow as tf

sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/dataset_prep.py')
import dataset_prep

sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/CVAE_Network.py')
import CVAE_Network

sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/Supporting_Function.py')
import Supporting_Function

dataset = ['fashion_mnist','grayscale']
# dataset = ['mnist','grayscale']
# dataset = ['emnist/letters','grayscale']
# dataset = ['sign_lang','grayscale']
# dataset = ['noise','grayscale']

# dataset = ['svhn_cropped','color']
# dataset = ['cifar10','color']
# dataset = ['gtsrb','color']
# dataset = ['celeb_a','color']
# dataset = ['compcars','color']
# dataset = ['kitti','color']
  
  
def Model_Training(argv):
    
  ##### Normalization 
  Normalization_Type = 'contrast_norm'
  #Normalization_Type = 'same' 
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
  Visible_Distribution = 'cont_bernoulli'
  
  print("Name of the Dataset:",str(dataset_name))
  print("Batch Size:",Batch_Size)
  print("Distribution:",Visible_Distribution)
  print("Number of Filters:",Number_of_Filters)
  print("Number of Channels:",Channels)
  print(" ")
                                        
  ##### Fetching Training, Validation & Testing Dataset                                    
  Training, Validation, Testing = dataset_prep.get_dataset(str(dataset_name), Batch_Size,mode,normalize=None,dequantize=False,shuffle_train=True,visible_dist=Visible_Distribution)
  
  ##### Fetching Variational Autoencoder Model
  Variational_Autoencoder = CVAE_Network.CVAE(input_shape=(Number_of_Filters, Number_of_Filters, Channels),num_filters=Number_of_Filters,latent_dim=Latent_Dimension,visible_dist=Visible_Distribution)
 
  ##### Optimizing & Compiling to imrpove the Model's Accuracy and reduce the Loss
  Directory_Model = os.path.join('test' , f'Models\{dataset_name}\{mode}\{Visible_Distribution}')
  checkpoint_path = os.path.join(Directory_Model, 'weights.hdf5')
  
  Optimizer = tf.optimizers.Adam(learning_rate=5e-4)
  Variational_Autoencoder.compile(optimizer=Optimizer,loss={'KL Divergence Loss': Variational_Autoencoder.kl_divergence_loss,'Decoder Loss': Variational_Autoencoder.decoder_nll_loss})
  Variational_Autoencoder.save_weights(filepath=checkpoint_path)
  
  ##### Initializing the Training & Evaluation
  
  Epochs=1# 50

  print("----------Initializing the Training----------")
  # Creating a callback to save the model's weights only during training
  callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
  Variational_Autoencoder.fit(Training,epochs=Epochs,validation_data=Validation,callbacks=[callback])
    
  print("----------Initializing the Evaluation----------")
  #weights = tf.io.gfile.listdir(Directory_Model).sort()
  Variational_Autoencoder.build([None]+list((Number_of_Filters, Number_of_Filters, Channels)))
    
  Variational_Autoencoder.load_weights(checkpoint_path)
    
  Variational_Autoencoder.compute_corrections(Training)
    
  Variational_Autoencoder.evaluate(Validation)
  probs_res = Supporting_Function.get_probs(dataset_name, Variational_Autoencoder ,mode,Normalization_Type, n_samples=5,split='test',training=False,visible_dist=Visible_Distribution)
                      
  with tf.io.gfile.GFile(os.path.join(Directory_Model, 'probs.pkl'), 'wb') as f:
     pickle.dump(probs_res, f)

if __name__ == '__main__':
  app.run(Model_Training)