# import all the necessary libraries
import collections
import numpy as np
import scipy
import scipy.interpolate
import tensorflow as tf
import tqdm
import Supporting_Function as utils


# def append_ll_to_dict(image, pixel_ll, probs_img):
#     '''It comes in Nd arrays (there are three channels, we need to flatten them) 
#     For example, we have in images a image of (32,32,3), and the probs for every pixel (32,32,3)
#     We go though every pixel and append the pixel value and its log-likelihood to the dictionary'''
#     for pix, pix_likelihood in zip(np.array(image).flatten(), np.array(pixel_ll).flatten()):
#         probs_img[int(pix)].append(pix_likelihood)
#     return probs_img

def append_ll_to_dict(images, pixel_ll, probs_img):
    '''It comes in Nd arrays (there are three channels, we need to flatten them) 
    For example, we have in images a image of (32,32,3), and the probs for every pixel (32,32,3)
    We go though every pixel and append the pixel value and its log-likelihood to the dictionary'''
    for pix, pix_likelihood in zip(np.array(images).flatten(), np.array(pixel_ll).flatten()):
        probs_img[int(pix)].append(pix_likelihood)
    return probs_img

def obtain_data_from_batch(train_batch, inp_shape):
    # Get targets data| inp = test_batch[0] | target = test_batch[1]
    data = train_batch[1].numpy()
    # if the pixel values are between 0 and 1, multiply by 255 and convert to int
    if data.max() <= 1:
        data = (data*255).astype(np.int32)
    if inp_shape[-1] == 3:
        data[:, :, 1:] += 256
        data[:, :, 2:] += 256
    num_channels = data.shape[-1]
    return data, num_channels

def interpolate_missing_values(corr_probs, num_channels):
    x = []
    y = []
    for pix, corr in corr_probs.items():
        x.append(pix)
        y.append(corr)
    f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
    missing_pixels = set(range(256*num_channels)) - set(corr_probs)
    for missing_pix in missing_pixels:
        corr_probs[missing_pix] = f(missing_pix)
    return corr_probs

def show_histogram(corr_probs):
    print('largest key of corr_probs: ', max(corr_probs.keys()))  
    for i in range(max(corr_probs.keys())):
      print(i, corr_probs[i])

    import matplotlib.pyplot as plt
    plt.hist(corr_probs.values(), bins=100)
    plt.show()

def compute_algorithmic_correction(model,inp_shape,dataset=None):
    """
    Compute correction for Categorical visible distribution.
    """
    # This is the dictionary with the pixel values as keys and the log-likelihoods as values
    probs_img = collections.defaultdict(list)
    # This is the dictionary with the pixel values as keys and the corrected log-likelihoods as values
    corr_probs = collections.defaultdict(list)

    # Go through every batch
    for batch_number,train_batch in enumerate(tqdm.tqdm(dataset)):
      # Get pixel log-likelihoods
      pixel_ll = utils.get_pix_ll(train_batch,model)
      
      # Get target data (images with shape [n_batch,32,32,n_channels]) and put pixels in range 0-255
      images,num_channels = obtain_data_from_batch(train_batch, inp_shape)

      # Loop over all images, channels and pixel values, and append to probs_img  the pixel values and their log-likelihoods
      probs_img = append_ll_to_dict(images, pixel_ll, probs_img)
      
      # After 500 batches, break the loop
      if batch_number == 500:
        break

    # Calculate corrections for each pixel value and channel
    for pixel_value in range(int(256*num_channels)):
      if len(probs_img[pixel_value]) > 0:
        log_sum_exp = tf.reduce_logsumexp(probs_img[pixel_value])
        log_count = tf.math.log(float(len(probs_img[pixel_value])))
        corrected_value = (log_sum_exp -  log_count).numpy()
        # print('pixel_value: ', pixel_value, ' corrected_value: ', corrected_value)
        corr_probs[int(pixel_value)] = corrected_value

    # Interpolate missing pixel values using linear interpolation
    corr_probs = interpolate_missing_values(corr_probs, num_channels)

    # # show a histogram of the corrections
    # show_histogram(corr_probs)

    # Create a function correct that takes a single argument x and returns the value of corr_probs[x]
    corr_probs_vectorize = np.vectorize(lambda x: corr_probs[x])
    return corr_probs_vectorize



  # def compute_algorithmic_correction(self, dataset=None):
  #   """
  #   Compute correction for Categorical visible distribution.
  #   """
  #   def append_ll_to_dict(data, pixel_ll, corr_dict_img):
  #     '''update the dictionary with the pixel values as keys and the log-likelihoods as values7'''
  #     # It comes in Nd arrays (because there are three channels, so we need to flatten them)
  #     for pix, pix_likelihood in zip(np.array(data).flatten(), np.array(pixel_ll).flatten()):
  #       corr_dict_img[int(pix)].append(pix_likelihood)

  #   def update_dict(corr_dict_img):
  #     '''update the dictionary with the pixel values as keys and the corrected log-likelihoods as values'''
  #     for pix in corr_dict_img:
  #       self.corr_dict[pix].append(scipy.special.logsumexp(corr_dict_img[pix])-np.log(len(corr_dict_img[pix])))
  
  #   # This is the dictionary with the pixel values as keys and the log-likelihoods as values
  #   corr_dict_img = collections.defaultdict(list)
  #   # This is the dictionary with the pixel values as keys and the corrected log-likelihoods as values
  #   self.corr_dict = collections.defaultdict(list)

  #   for batch_number,train_batch in enumerate(tqdm.tqdm(dataset)):
  #     # Get pixel log-likelihoods
  #     pixel_ll = utils.get_pix_ll(train_batch, self)
      
  #     # Get targets data. inp = test_batch[0], target = test_batch[1]
  #     data = train_batch[1].numpy()
  #     # if the pixel values are between 0 and 1, multiply by 255 and convert to int
  #     if data.max() <= 1:
  #         data = (data*255).astype(np.int32)
  #     if self.inp_shape[-1] == 3:
  #         data[:, :, 1:] += 256
  #         data[:, :, 2:] += 256
  #     num_channels = data.shape[-1]

  #     # Iterate over images in batch
  #     for i,image in enumerate(data):
  #       # Loop over all channels and pixel values, and append to corr_dict_img  the pixel values and their log-likelihoods
  #       append_ll_to_dict(data[i], pixel_ll[i], corr_dict_img)
  #       # I dont need this, because after the loop I update the dictionary with the corrected log-likelihoods
  #       # update_dict(corr_dict_img)

  #     if batch_number == 500:
  #       break

  #   # Calculate corrections for each pixel value and channel
  #   for pixel_value in range(int(256*num_channels)):
  #     if len(corr_dict_img[pixel_value]) > 0:
  #       log_sum_exp = tf.reduce_logsumexp(corr_dict_img[pixel_value])
  #       log_count = tf.math.log(float(len(corr_dict_img[pixel_value])))
  #       corrected_value = (log_sum_exp -  log_count).numpy()
  #       # print('pixel_value: ', pixel_value, ' corrected_value: ', corrected_value)
  #       self.corr_dict[int(pixel_value)] = corrected_value

  #    # Interpolate missing pixel values using linear interpolation
  #   x = []
  #   y = []
  #   for pix, corr in self.corr_dict.items():
  #       x.append(pix)
  #       y.append(corr)
  #   f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
  #   missing_pixels = set(range(256*num_channels)) - set(self.corr_dict)
  #   for missing_pix in missing_pixels:
  #       self.corr_dict[missing_pix] = f(missing_pix)
    
  #   # print('largest key of self.corr_dict: ', max(self.corr_dict.keys()))  
  #   # for i in range(max(self.corr_dict.keys())):
  #   #   print(i, self.corr_dict[i])

  #   # # show a histogram of the corrections
  #   # import matplotlib.pyplot as plt
  #   # plt.hist(self.corr_dict.values(), bins=100)
  #   # plt.show()

  #   # Create a function correct that takes a single argument x and returns the value of self.corr_dict[x]
  #   self.correct = np.vectorize(lambda x: self.corr_dict[x])

