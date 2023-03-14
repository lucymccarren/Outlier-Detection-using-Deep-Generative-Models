import collections

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.special
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import time

import sys
# sys.path.append('C:/Users/Shahnawaz/Desktop/DD2412 DL Advanced/vae_ood/Compute_Likelihood_Analytically.py')
import _1_analytical_correction
import Supporting_Function  as utils
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VAE(tfk.Model):
  """Generic Variational Autoencoder."""

  def __init__(self,input_shape,latent_dim,visible_dist,encoder,decoder):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    self.inp_shape = input_shape
    self.visible_dist = visible_dist
    self.latent_prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim),scale_diag=tf.ones(self.latent_dim))
    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs, training=False):
    self.posterior = self.encoder(inputs, training=training)
    self.code = self.posterior.sample()
    self.decoder_likelihood = self.decoder(self.code, training=training)
    return {'posterior': self.posterior, 'decoder_ll': self.decoder_likelihood}
  

  def compute_algorithmic_correction(self, dataset=None):
    """
    Compute correction for Categorical visible distribution.
    """
    def append_ll_to_dict(data, pixel_ll, corr_dict_img):
      '''update the dictionary with the pixel values as keys and the log-likelihoods as values7'''
      # It comes in Nd arrays (because there are three channels, so we need to flatten them)
      for pix, pix_likelihood in zip(np.array(data).flatten(), np.array(pixel_ll).flatten()):
        corr_dict_img[int(pix)].append(pix_likelihood)

    def update_dict(corr_dict_img):
      '''update the dictionary with the pixel values as keys and the corrected log-likelihoods as values'''
      for pix in corr_dict_img:
        self.corr_dict[pix].append(scipy.special.logsumexp(corr_dict_img[pix])-np.log(len(corr_dict_img[pix])))
  
    # This is the dictionary with the pixel values as keys and the log-likelihoods as values
    corr_dict_img = collections.defaultdict(list)
    # This is the dictionary with the pixel values as keys and the corrected log-likelihoods as values
    self.corr_dict = collections.defaultdict(list)

    for batch_number,train_batch in enumerate(tqdm.tqdm(dataset)):
      # Get pixel log-likelihoods
      pixel_ll = utils.get_pix_ll(train_batch, self)
      
      # Get targets data. inp = test_batch[0], target = test_batch[1]
      data = train_batch[1].numpy()
      # if the pixel values are between 0 and 1, multiply by 255 and convert to int
      if data.max() <= 1:
          data = (data*255).astype(np.int32)
      if self.inp_shape[-1] == 3:
          data[:, :, 1:] += 256
          data[:, :, 2:] += 256
      num_channels = data.shape[-1]

      # Iterate over images in batch
      for i,image in enumerate(data):
        # Loop over all channels and pixel values, and append to corr_dict_img  the pixel values and their log-likelihoods
        append_ll_to_dict(data[i], pixel_ll[i], corr_dict_img)
        # I dont need this, because after the loop I update the dictionary with the corrected log-likelihoods
        # update_dict(corr_dict_img)

      if batch_number == 10:
        break

    # Calculate corrections for each pixel value and channel
    for pixel_value in range(int(256*num_channels)):
      if len(corr_dict_img[pixel_value]) > 0:
        log_sum_exp = tf.reduce_logsumexp(corr_dict_img[pixel_value])
        log_count = tf.math.log(float(len(corr_dict_img[pixel_value])))
        corrected_value = (log_sum_exp -  log_count).numpy()
        # print('pixel_value: ', pixel_value, ' corrected_value: ', corrected_value)
        self.corr_dict[int(pixel_value)] = corrected_value

     # Interpolate missing pixel values using linear interpolation
    x = []
    y = []
    for pix, corr in self.corr_dict.items():
        x.append(pix)
        y.append(corr)
    f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
    missing_pixels = set(range(256*num_channels)) - set(self.corr_dict)
    for missing_pix in missing_pixels:
        self.corr_dict[missing_pix] = f(missing_pix)
    
    # print('largest key of self.corr_dict: ', max(self.corr_dict.keys()))  
    # for i in range(max(self.corr_dict.keys())):
    #   print(i, self.corr_dict[i])

    # # show a histogram of the corrections
    # import matplotlib.pyplot as plt
    # plt.hist(self.corr_dict.values(), bins=100)
    # plt.show()

    # Create a function correct that takes a single argument x and returns the value of self.corr_dict[x]
    self.correct = np.vectorize(lambda x: self.corr_dict[x])


  def compute_corrections(self, dataset=None):
    # pylint: disable=g-long-lambda
    if self.visible_dist == 'cont_bernoulli':
      self.corr_dict = {}
      targets = np.linspace(1e-3, 1-1e-3, 999)
      for target in targets:
        self.corr_dict[(target * 1000).round().astype(np.int32)] = _1_analytical_correction.Neg_Reconstruction_Error(
                        scipy.optimize.fmin(_1_analytical_correction.Neg_Reconstruction_Error, 0.5, args=(target,), disp=False)[0],target)
      corr_func = lambda pix: self.corr_dict[pix]
      self.correct = np.vectorize(corr_func)
      
    elif self.visible_dist == 'bernoulli':
      self.corr_dict = dict(zip(np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3),
          tfd.Bernoulli(probs=tf.linspace(1e-3, 1-1e-3, 999)).log_prob(tf.linspace(1e-3, 1-1e-3, 999)).numpy()))
      corr_func = lambda pix: self.corr_dict[(np.clip(pix, 1e-3, 1-1e-3).astype(float).round(decimals=3))].astype(np.float32)
      self.correct = np.vectorize(corr_func)
    elif self.visible_dist == 'gaussian':
      assert dataset is not None, ('dataset is required to compute correction '
                                   'for Gaussian visible distribution.')
      self.corr_dict = collections.defaultdict(list)
      update_dict = lambda corr_dict_img: [self.corr_dict[pix].append(scipy.special.logsumexp(corr_dict_img[pix]) - np.log(len(corr_dict_img[pix]))) for pix in corr_dict_img]
      j = 0
      for train_batch in tqdm.tqdm(dataset):
        j += 1
        pixel_ll = _1_analytical_correction.Pixel_Likelihood(train_batch, self)
        inp = train_batch[1].numpy()
        if self.inp_shape[-1] == 3:
          inp[:, :, :, 1:] += 1
          inp[:, :, :, 2:] += 1
        for i in range(inp.shape[0]):
          corr_dict_img = collections.defaultdict(list)
          # pylint: disable=cell-var-from-loop
          np.vectorize(lambda pix, ll: corr_dict_img[np.round(pix, decimals=2)].append(ll),otypes=[float])(inp[i], pixel_ll[i])
          update_dict(corr_dict_img)

        if j == 500:
          break

      for key in self.corr_dict:
        n = len(self.corr_dict[key])
        self.corr_dict[key] = scipy.special.logsumexp(np.array(self.corr_dict[key])) - np.log(n)

      f = scipy.interpolate.interp1d(*list(zip(*[(pix, corr) for pix, corr in self.corr_dict.items()])), fill_value='extrapolate')
      for missing_pix in (set(np.round(np.linspace(0, 3, 301), 2)) - set(self.corr_dict)):
        self.corr_dict[missing_pix] = f(missing_pix)
      self.correct = np.vectorize(lambda x: self.corr_dict[np.round(x, 2)])
    elif self.visible_dist == 'categorical':
      self.compute_algorithmic_correction(dataset=dataset)

  def kl_divergence_loss(self, target, posterior):
    kld = tfd.kl_divergence(posterior, self.latent_prior)
    return tf.reduce_mean(kld)

  def decoder_nll_loss(self, target, decoder_likelihood):
    decoder_nll = -(decoder_likelihood.log_prob(target))
    decoder_nll = tf.reduce_sum(decoder_nll, axis=[1, 2, 3])
    return tf.reduce_mean(decoder_nll)

  def log_prob(self, inp, target, n_samples, training=False):
    """Computes an importance weighted log likelihood estimate."""
    posterior = self.encoder(inp, training=training)
    code = posterior.sample(n_samples)
    kld = posterior.log_prob(code) - self.latent_prior.log_prob(code)
    visible_dist = self.decoder(tf.reshape(code, [-1, self.latent_dim]),training=training)
    target_rep = tf.reshape(
        tf.repeat(tf.expand_dims(target, 0), n_samples, 0),[-1] + list(self.inp_shape))
    decoder_ll = visible_dist.log_prob(target_rep)
    decoder_ll = tf.reshape(decoder_ll, [n_samples, -1] + list(self.inp_shape))
    decoder_ll = tf.reduce_sum(decoder_ll, axis=[2, 3, 4])

    elbo = tf.reduce_logsumexp(decoder_ll - kld, axis=0)
    elbo = elbo - tf.math.log(tf.cast(n_samples, dtype=tf.float32))
    return elbo


class CVAE(VAE):
  """Convolutional Variational Autoencoder."""

  def __init__(self, input_shape, num_filters, latent_dim, visible_dist):
    num_channels = input_shape[-1]
    encoder = tfk.Sequential([tfkl.InputLayer(input_shape=input_shape),
            tfkl.Conv2D(filters=num_filters, kernel_size=4, strides=2,padding='SAME'),tfkl.BatchNormalization(),tfkl.ReLU(),
            tfkl.Conv2D(filters=2*num_filters, kernel_size=4, strides=2,padding='SAME'),tfkl.BatchNormalization(),tfkl.ReLU(),
            tfkl.Conv2D(filters=4*num_filters, kernel_size=4, strides=2,padding='SAME'),tfkl.BatchNormalization(),tfkl.ReLU(),
            tfkl.Conv2D(filters=2*latent_dim, kernel_size=4, strides=1,padding='VALID'),tfkl.Flatten(),
            tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[Ellipsis, :latent_dim],scale_diag=tf.nn.softplus(t[Ellipsis, latent_dim:])))])

    decoder_head = []
    if visible_dist == 'cont_bernoulli':
      decoder_head.append(tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,strides=2, padding='SAME'))
      decoder_head.append(tfpl.DistributionLambda(lambda t: tfd.ContinuousBernoulli(logits=tf.clip_by_value(t, -15.94, 15.94), validate_args=True), convert_to_tensor_fn=lambda s: s.logits))
    if visible_dist == 'bernoulli':
      decoder_head.append(tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,strides=2, padding='SAME'))
      decoder_head.append(tfpl.DistributionLambda(lambda t: tfd.Bernoulli(logits=tf.clip_by_value(t, -15.94, 15.94), validate_args=False), convert_to_tensor_fn=lambda s: s.logits))
    elif visible_dist == 'gaussian':
      decoder_head.append(tfkl.Conv2DTranspose(filters=num_channels, kernel_size=4,strides=2, padding='SAME', activation='sigmoid'))
      decoder_head.append(tfpl.DistributionLambda(lambda t: tfd.TruncatedNormal(loc=t, scale=0.2, low=0, high=1,), convert_to_tensor_fn=lambda s: s.loc))
    elif visible_dist == 'categorical':
      decoder_head.append(tfkl.Conv2DTranspose(filters=num_channels*256, kernel_size=4,strides=2, padding='SAME'))
      decoder_head.append(tfkl.Reshape(list(input_shape) + [256]))
      decoder_head.append(tfpl.DistributionLambda(lambda t: tfd.Categorical(logits=t, validate_args=True), convert_to_tensor_fn=lambda s: s.logits))

    decoder = tfk.Sequential([tfkl.InputLayer(input_shape=(latent_dim,)),
            tfkl.Reshape([1, 1, latent_dim]),
            tfkl.Conv2DTranspose(filters=4*num_filters, kernel_size=4,strides=1, padding='VALID'),
            tfkl.BatchNormalization(),tfkl.ReLU(),
            tfkl.Conv2DTranspose(filters=2*num_filters, kernel_size=4,strides=2, padding='SAME'),
            tfkl.BatchNormalization(),tfkl.ReLU(),
            tfkl.Conv2DTranspose(filters=num_filters, kernel_size=4,strides=2, padding='SAME'),
            tfkl.BatchNormalization(),tfkl.ReLU(),*decoder_head])

    super(CVAE, self).__init__(input_shape,latent_dim,visible_dist,encoder,decoder)
