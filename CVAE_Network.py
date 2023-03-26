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
from _2_algorithmic_correction import compute_algorithmic_correction
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
  

  def compute_corrections(self, dataset=None):
    # pylint: disable=g-long-lambda
    if self.visible_dist == 'cont_bernoulli':
      start=1e-3; stop=1-start; num=999
      target_pixels = np.linspace(start, stop, num)
      Reconstruction_LL = _1_analytical_correction.Analytical_Correction_For_Intensity_Bias(target_pixels)
      self.correct=np.vectorize(lambda x: Reconstruction_LL[x])
      
    elif self.visible_dist == 'bernoulli':
      self.corr_dict = dict(zip(np.round(np.linspace(1e-3, 1-1e-3, 999), decimals=3),
          tfd.Bernoulli(probs=tf.linspace(1e-3, 1-1e-3, 999)).log_prob(tf.linspace(1e-3, 1-1e-3, 999)).numpy()))
      print('self.corr_dict',self.corr_dict)
      corr_func = lambda pix: self.corr_dict[(np.clip(pix, 1e-3, 1-1e-3).astype(float).round(decimals=3))].astype(np.float32)
      self.correct = np.vectorize(corr_func)
      
    elif self.visible_dist == 'categorical':
      self.correct = compute_algorithmic_correction(self,self.inp_shape,dataset=dataset)

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
