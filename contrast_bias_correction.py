from skimage.transform import resize
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk

def contrast_stretch_pctile5(img):
    """Corresponds to formula given in part 3.3"""
    a = tfp.stats.percentile(img, 5)
    r = tfp.stats.percentile(img, 95) - a
    norm = (img - a) / r
    #any values less than 0 clip to 0, greater than 1 clip to 1
    norm = tf.clip_by_value(norm, 0., 1.)
    return norm

def contrast_stretch_pctile2(img):
    # Contrast stretching   
    p2, p98 = np.percentile(img, (2, 98))
    if (p2==p98):
        return img      
    # some images are just one color, so they gerenate an divide by zero error, so return original image
    img_contrast_stretch = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_contrast_stretch

def equalization(img):
    if (np.max(img) == np.min(img) ):
        return img      
    # Equalization
    img_equalized = exposure.equalize_hist(img)
    return img_equalized

def adaptive_equalization(img):
    if (np.max(img) == np.min(img) ):
        return img      
    # Adaptive Equalization
    img_adaptive_equalized = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adaptive_equalized

def local_equalization(img):
    if (np.max(img) == np.min(img) ):
        return img      
    # Local Equalization--for details see http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
    selem = disk(30)
    img_local_equal = rank.equalize(img, selem=selem)
    return img_local_equal