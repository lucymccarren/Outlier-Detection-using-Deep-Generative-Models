import os
import pickle

from absl import app
import tensorflow as tf
from vae_ood import Generate_Datasets, CVAE_Network, utils 

# Generate_Datasets: Dataset_Utils
# CVAE_Network: Network

def Model_Training(dataset):
    
  ##### Normalization
  Normalization_type = None
  Test_Normalization_type = 'same' # To be applied at test time

  if Test_Normalization_type == 'same':
    test_normalize = Normalization_type
  else:
    test_normalize = Test_Normalization_type
  
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
  Evaluation_Every = 100
  Number_of_Filters = 32
  Latent_Dimension = 100
  Visible_Distribution = 'cont_bernoulli'

  print("Batch Size:",Batch_Size)
  print("Distribution:",Visible_Distribution)
  print("Number of Filters:",Number_of_Filters)
  print(" ")
                                        
  ##### Fetching Training, Validation & Testing Dataset                                    
  Training, Validation, Testing = Generate_Datasets.get_dataset(dataset_name, Batch_Size,mode,normalize=Normalization_type,dequantize=False,visible_dist=Visible_Distribution)
  
  ##### Fetching Variational Autoencoder Model
  Variational_Autoencoder = CVAE_Network.CVAE(input_shape=(Number_of_Filters, Number_of_Filters, Channels),num_filters=Number_of_Filters,latent_dim=Latent_Dimension,visible_dist=Visible_Distribution)
 
  ##### Optimizing & Compiling to imrpove the Model's Accuracy and reduce the Loss
  Optimizer = tf.optimizers.Adam(learning_rate=5e-4)
  Variational_Autoencoder.compile(optimizer=Optimizer,loss={'KL Divergence Loss': Variational_Autoencoder.kl_divergence_loss,'Decoder Loss': Variational_Autoencoder.decoder_nll_loss})
  
  ##### Initializing the Training & Evaluation
  start_training= True
  start_evaluation = True
  Epochs=10
  Directory_Model = os.path.join('test' , f'Models/{dataset_name}/{mode}/{Channels}')
  Directory_Log = os.path.join('test' , 'logs',  f'Models/{dataset_name}/{mode}/{Channels}')

  if start_training:
    tf.io.gfile.makedirs(Directory_Model)
    callbacks = [utils.TensorBoardWithLLStats(Evaluation_Every, dataset_name,dataset_name, mode, Normalization_type, Visible_Distribution,log_dir=Directory_Log,
            update_freq='epoch'), tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(Directory_Model, 'weights.hdf5'),verbose=1, save_weights_only=True, save_best_only=True)]

    print("Initializing the Training")
    Variational_Autoencoder.fit(Training,epochs=Epochs,validation_data=Validation,callbacks=callbacks)
  
  if start_evaluation:
    weights = tf.io.gfile.listdir(Directory_Model).sort()
    Variational_Autoencoder.build([None]+list((Number_of_Filters, Number_of_Filters, Channels)))
    
    weights_path = os.path.join(Directory_Model, weights[-1])
    Variational_Autoencoder.load_weights(weights_path)

    Variational_Autoencoder.compute_corrections(Training)
    
    print("Initializing the Evaluation")
    Variational_Autoencoder.evaluate(Validation)
    probs_res = utils.get_probs(dataset_name, Variational_Autoencoder ,mode,test_normalize, n_samples=5,split='test',training=False,visible_dist=Visible_Distribution)
                      
    with tf.io.gfile.GFile(os.path.join(Directory_Model, 'probs.pkl'), 'wb') as f:
      pickle.dump(probs_res, f)

if __name__ == '__main__':
    
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
  
  app.run(Model_Training(dataset))