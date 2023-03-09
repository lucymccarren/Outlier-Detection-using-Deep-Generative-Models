def correction_contrast_bias(im):
    """Corresponds to formula given in part 3.3"""

    a = tfp.stats.percentile(image, 5)
    r = tfp.stats.percentile(image, 95) - a

    norm = (im - a) / r
    
    #any values less than 0 clip to 0, greater than 1 clip to 1
    norm = tf.clip_by_value(norm, 0., 1.)

    return norm
