import csv
import functools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from _3_contrast_bias_correction import *

## Data Augmentation
partial = functools.partial


def Generate_Vertically_Flipped_Cifar10(split, cls):
    image='cifar10'
    
    X = tf.placeholder(tf.float32, shape = (image.shape[0], image.shape[1], 3))
    flipped = tf.image.flip_up_down(X)
    
    (ds_train, ds_val, ds_test) = tfds.load(flipped ,split=['train[:90%]','train[90%:]','test'],as_supervised=True)
    ds_test = ds_test.shuffle(1000)

    if split == b'train':
        ds = ds_train
    elif split == b'val':
        ds = ds_val
    else:
        ds = ds_test

    for x, y in ds:
        if y == cls:
            yield x

def Generate_Vertically_Flipped_Kitti(split, cls):
    image='kitti'
    flipped = tf.image.flip_up_down(image)
    
    (ds_train, ds_val, ds_test) = tfds.load(flipped,split=['train[:90%]','train[90%:]','test'],as_supervised=True)
   
    if split == b'train':
        ds = ds_train
    elif split == b'val':
        ds = ds_val
    else:
        ds = ds_test

    for x, y in ds:
        if y == cls:
            yield x
            

    
def noise_generator(split=b'val', mode=b'grayscale'):
    if split == b'train':
        np.random.seed(0)
    if split == b'val':
        np.random.seed(1)
    else:
        np.random.seed(2)
    for _ in range(10000):
        if mode == b'grayscale':
            yield np.random.randint(low=0, high=256, size=(32, 32, 1))
        else:
            yield np.random.randint(low=0, high=256, size=(32, 32, 3))
                        
def image_contrast(split=b'train'):
    rootpath = 'datasets/image_contrast'
    
    random.seed(42)

    if split in [b'train', b'val']:
        for i in range(1,100):
            yield plt.imread(os.path.join(rootpath,'Train', 'greta_'+str(i)+'.png'))

    elif split == b'test':
        for i in range(1,100):
            yield plt.imread(os.path.join(rootpath, 'Test', 'greta_'+str(i)+'.png')) 

def pixel_intensity(split=b'train'):
    random.seed(42)
    
    rootpath = 'datasets/pixel_intensity'
    
    if split in [b'train', b'val']:
        for i in range(1,254): 
            yield plt.imread(os.path.join(rootpath,'Train',str(i)+'.png'))

    elif split == b'test':
        for i in range(1,254): 
            #print('.\\vae_ood\\datasets\\pixel_intensity\\Test\\'+str(i)+'.png') 
            yield plt.imread(os.path.join(rootpath,'Test',str(i)+'.png'))
            

def hand_sign_mnist_builder():
    """Generator function for the grayscale Hand Sign MNIST dataest.
    Source: https://www.kaggle.com/ash2703/handsignimages
    Returns:
    A dataset builder object
    """
    rootpath = 'datasets/SignLang'

    random.seed(42)
    return tfds.folder_dataset.ImageFolder(rootpath)

def gtsrb_generator(split=b'train'):
    """Generator function for the GTSRB Dataset.
    Source: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
    Args:
    split: Data split to load - "train", "val" or "test".
    Yields:An image
    """

    rootpath = 'datasets/GTSRB'
    random.seed(42)
    # random.seed(43) # split 2

    if split in [b'train', b'val']:
        rootpath = os.path.join(rootpath, 'Final_Training', 'Images')
        all_images = []
        for c in range(0, 43):
            prefix = rootpath + '/' + format(c, '05d') + '/'
            gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
            gt_reader = csv.reader(gt_file, delimiter=';')
            next(gt_reader)

            for row in gt_reader:
                all_images.append((prefix + row[0],
                                   (int(row[3]), int(row[4]), int(row[5]),int(row[6]))
                                  ))
            gt_file.close()
        random.shuffle(all_images)
        if split == b'train':
            all_images = all_images[:-(len(all_images)//10)]
        else:
            all_images = all_images[-(len(all_images)//10):]
        for image, _ in all_images:
            img = plt.imread(image)
            yield img

    elif split == b'test':
        rootpath = os.path.join(rootpath, 'Final_Test', 'Images/')
        gt_file = open(rootpath + '/GT-final_test.test.csv')
        gt_reader = csv.reader(gt_file, delimiter=';')
        next(gt_reader)
        for row in gt_reader:
            img = plt.imread(rootpath + row[0])
            yield img
        gt_file.close()


def cifar10_class_generator(split, cls):
    """Generator function of class wise CIFAR10 dataset.
    Args:
    split: Data split to load - "train", "val" or "test".
    cls: The target class to load examples from.
    Yields: An image
    """
    (ds_train, ds_val, ds_test) = tfds.load('cifar10',split=['train[:90%]','train[90%:]','test'],as_supervised=True)

    if split == b'train':
        ds = ds_train
    elif split == b'val':
        ds = ds_val
    else:
        ds = ds_test

    for x, y in ds:
        if y == cls:
            yield x

def kitti(split, cls):
    (ds_train, ds_val, ds_test) = tfds.load('kitti',split=['train[:90%]','train[90%:]','test'],as_supervised=True)
   
    if split == b'train':
        ds = ds_train
    elif split == b'val':
        ds = ds_val
    else:
        ds = ds_test

    for x, y in ds:
        if y == cls:
            yield x
            

def get_dataset(name,batch_size,mode,normalize=None, dequantize=False,shuffle_train=True,visible_dist='cont_bernoulli' ):
    print('Loading dataset: ', name)
    def preprocess(image, inverted, mode, normalize, dequantize, visible_dist):
        if isinstance(image, dict):
            image = image['image']
        image = tf.cast(image, tf.float32)
        if dequantize:
            image += tf.random.uniform(image.shape)
            image = image / 256.0
        else:
            image = image / 255.0
        image = tf.image.resize(image, [32, 32], antialias=True)
        if mode == 'grayscale':
            if image.shape[-1] != 1:
                image = tf.image.rgb_to_grayscale(image)
        else:
            if image.shape[-1] != 3:
                image = tf.image.grayscale_to_rgb(image)

        if isinstance(normalize, str) and normalize.startswith('contrast_stretch_pctile5'):
            
            contrast_stretch_pctile5(image)
         
        elif normalize == 'equalization':
            
            equalization(image)
          
        elif normalize == 'adaptive_equalization':
            
            adaptive_equalization(image)

        elif normalize is not None:
            
            raise NotImplementedError(f'Normalization method {normalize} not implemented')

        if inverted:
            image = 1 - image
        image = tf.clip_by_value(image, 0., 1.)

        target = image
        if visible_dist == 'categorical':
            target = tf.round(target*255)

        return image, target

    assert name in ['fashion_mnist',
      'svhn_cropped', 'cifar10', 'celeb_a', 'gtsrb', 'compcars', 'mnist','kitti', 'sign_lang', 'emnist/letters', 'noise','image_contrast','pixel_intensity',
      *[f'cifar10-{i}' for i in range(10)]], f'Dataset {name} not supported'

    inverted = False
    if name.endswith('inverted'):
        name = name[:-9]
        inverted = True

    if name == 'noise':
        n_channels = 1 if mode == 'grayscale' else 3
        ds_train = tf.data.Dataset.from_generator(noise_generator,args=['train', mode],output_types=tf.int32,output_shapes=(None, None, n_channels))
        ds_val = tf.data.Dataset.from_generator(noise_generator,args=['val', mode],output_types=tf.int32,output_shapes=(None, None, n_channels))
        ds_test = tf.data.Dataset.from_generator(noise_generator,args=['test', mode],output_types=tf.int32,output_shapes=(None, None, n_channels))
        n_examples = 1024
    
    elif name.startswith('gtsrb'):
        ds_train = tf.data.Dataset.from_generator(gtsrb_generator,args=['train'],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_val = tf.data.Dataset.from_generator(gtsrb_generator,args=['val'],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_test = tf.data.Dataset.from_generator(gtsrb_generator,args=['test'],output_types=tf.int32,output_shapes=(None, None, 3))
        n_examples = 1024
    
    elif name.startswith('kitti-'):
        n_examples = 7481
        cls = int(name.split('-')[1])
        ds_train = tf.data.Dataset.from_generator(kitti,args=['train', cls],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_val = tf.data.Dataset.from_generator(kitti,args=['val', cls],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_test = tf.data.Dataset.from_generator(kitti,args=['test', cls],output_types=tf.int32,output_shapes=(None, None, 3))
    
    elif name == 'pixel_intensity':
        ds_train = tf.data.Dataset.from_generator(pixel_intensity,args=['train'],output_types=tf.int32,output_shapes=(None, None, 3))
        print(ds_train)
        ds_val = tf.data.Dataset.from_generator(pixel_intensity,args=['val'],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_test = tf.data.Dataset.from_generator(pixel_intensity,args=['test'],output_types=tf.int32,output_shapes=(None, None, 3))
        n_examples = 11        

    elif name == 'image_contrast':
        ds_train = tf.data.Dataset.from_generator(image_contrast,args=['train'],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_val = tf.data.Dataset.from_generator(image_contrast,args=['val'],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_test = tf.data.Dataset.from_generator(image_contrast,args=['test'],output_types=tf.int32,output_shapes=(None, None, 3))
        n_examples = 10   
        
    elif name.startswith('cifar10-'):
        n_examples = 1024
        cls = int(name.split('-')[1])
        ds_train = tf.data.Dataset.from_generator(cifar10_class_generator,args=['train', cls],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_val = tf.data.Dataset.from_generator(cifar10_class_generator,args=['val', cls],output_types=tf.int32,output_shapes=(None, None, 3))
        ds_test = tf.data.Dataset.from_generator(cifar10_class_generator,args=['test', cls],output_types=tf.int32,output_shapes=(None, None, 3))
    
    elif name == 'sign_lang':
        builder = hand_sign_mnist_builder()
        ds_train = builder.as_dataset('Train')
        ds_val = builder.as_dataset('Val')
        ds_test = builder.as_dataset('Test')
        n_examples = 1024

    else:
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
        name, split=['train[:90%]', 'train[90%:]', 'test'], with_info=True)
        n_examples = ds_info.splits['train'].num_examples

    ds_train = ds_train.map(
        partial(preprocess, inverted=inverted, mode=mode,
                normalize=normalize, dequantize=dequantize,
                visible_dist=visible_dist,),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    if shuffle_train:
        ds_train = ds_train.shuffle(n_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.map(
        partial(preprocess, inverted=inverted, mode=mode,
                normalize=normalize, dequantize=dequantize,
                visible_dist=visible_dist,),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        partial(preprocess, inverted=inverted, mode=mode,
                normalize=normalize, dequantize=dequantize,
                visible_dist=visible_dist,),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test
