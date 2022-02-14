""" Transforms Factory

Adapted from TIMM / Ross Wightman:
https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/data/transforms_factory.py#L44

"""
import math

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

# from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform

# use custom version for cv2 instead of PIL
# from dataloaders.autoaugment.autoaug_cv2 import rand_augment_transform, augment_and_mix_transform, auto_augment_transform

from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
from timm.data.random_erasing import RandomErasing

import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np

from numba import njit
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.writer import DatasetWriter



""" AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and improvements and
does not include any of the search code.
AA and RA Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data - https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection - https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty - https://arxiv.org/abs/1912.02781
Hacked together by / Copyright 2019, Ross Wightman
"""


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

# def rotate(img, degrees, **kwargs)
#     pass
def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


# swapping np functions instead of PIL
def auto_contrast(img, prob, level_args, fillcolor, resample):
    """Implements Autocontrast function from PIL using numpy ops instead.
    Args:
    image: A 3D uint8 tensor.
    Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.

    source:
    https://github.com/poodarchu/learn_aug_for_object_detection.numpy/blob/master/autoaugment_utils.py
    """

    # no op
    if prob < 1.0 and random.random() > prob:
        return

    def scale_channel(img):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = float(np.min(img))
        hi = float(np.max(img))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            im = np.clip(im, 0.0, 255.0)
            return im.astype(np.uint8)

        if hi > lo:
            result = scale_values(img)
        else:
            result = img

        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    img[:,:, 0] = scale_channel(img[:, :, 0])
    img[:,:, 1] = scale_channel(img[:, :, 1])
    img[:,:, 2] = scale_channel(img[:, :, 2])
    # img = np.stack((s1, s2, s3), 2)
    # return img

def equalize(img, prob, level_args, fillcolor, resample):
    
    """
    
    source / adapted from these:
    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
    and
    https://msameeruddin.hashnode.dev/image-equalization-contrast-enhancing-in-python

    note - can't use np.ma, not numba supported

    """

    # no op
    if prob < 1.0 and random.random() > prob:
        return

    image_flattened = img.flatten()

    bins_const = 256
    image_hist,bins = np.histogram(image_flattened,bins_const,[0,bins_const])

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape

    img[:,:,:] = image_eq.reshape(img.shape)
    return 

def invert(img, prob, level_args, fillcolor, resample):
    """
    Source:
    https://yashaslokesh.github.io/inverting-the-colors-of-an-image.html

    """

    # no op
    if prob < 1.0 and random.random() > prob:
        return

    img[:,:,:] = 255 - img

    # return 255 - img

def solarize(img, prob, level_args, fillcolor, resample):
    # no op
    if prob < 1.0 and random.random() > prob:
        return

    thresh = level_args[0]
    img[:,:,:] = np.where(img < thresh, img, 255 - img).astype(np.uint8)  # make sure to cast!

    # img = new_img
    # return new_img
    
# def invert(img, **__):
#     return ImageOps.invert(img)

# def equalize(img, **__):
#     return ImageOps.equalize(img)


# def solarize(img, thresh, **__):
#     return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)

# def color(img, factor, **__):
#     return ImageEnhance.Color(img).enhance(factor)

def color(img, prob, level_args, fillcolor, resample):
    """

    def color(img, factor):
        # Equivalent of PIL Color
        if img.shape[0] == 0 or img.shape[1] == 0:
            return img

        degenerate = np.tile(rgb2gray(img)[..., np.newaxis], [1, 1, 3])
        return blend(degenerate, img, factor)

    """

    # no op
    if prob < 1.0 and random.random() > prob:
        return
    if img.shape[0] == 0 or img.shape[1] == 0:
        return

    factor = level_args[0]

    degenerate_gray = np.expand_dims(_rgb2gray(img), axis=2)
    stacked_gray = np.concatenate((degenerate_gray, degenerate_gray, degenerate_gray), axis=2)
    img[:,:,:] = _blend(stacked_gray, img, factor)
    return

@njit
def _rgb2gray(img):
    """Helpers, note it returns an image"""
    # return np.dot(img[...,:3], np.array([0.2989, 0.5870, 0.1140])).astype(np.uint8)

    h, w, c = img.shape

    ch = (0.2989, 0.5870, 0.1140)

    a = img[:,:,0].flatten() * ch[0]
    b = img[:,:,1].flatten() * ch[1]
    c = img[:,:,2].flatten() * ch[2]

    d = a + b + c

    return d.astype(np.uint8).reshape(h, w)

    # return img.dot([0.2989, 0.5870, 0.1140]).astype(np.uint)

@njit
def _blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.
    Returns:
    A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return image1.astype(np.uint8)
    if factor == 1.0:
        return image2.astype(np.uint8)

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
        return temp.astype(np.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return np.clip(temp, 0.0, 255.0).astype(np.uint8)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * .9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _LEVEL_DENOM) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _LEVEL_DENOM) * 110),


# not sure what this maps to
LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,  # before had Auto_Contrast
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


# this needs to be a function that returns numpy functions, and passes the right hyparams
# needs to do the exact same thing as Augment Op basically
def augment_op_factory(name, prob=0.5, magnitude=10, hparams=None):

    op_name = name
    # note it uses both name to op and level to arg (but level is sometimes None)
    op_aug_fn = NAME_TO_OP[op_name]
    op_level_fn = LEVEL_TO_ARG[op_name]
    op_prob = prob
    op_magnitude = magnitude
    op_hparams = hparams.copy()

    fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
    fillcolor_np = np.array(fillcolor)
    
    resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
    resample_np = np.array(resample)

    op_magnitude_std = op_hparams.get('magnitude_std', 0)
    op_magnitude_max = op_hparams.get('magnitude_max', None)

    if op_magnitude_std > 0:
        # magnitude randomization enabled
        if op_magnitude_std == float('inf'):
            op_magnitude = random.uniform(0, op_magnitude)
        elif op_magnitude_std > 0:
            op_magnitude = random.gauss(op_magnitude, op_magnitude_std)
    # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
    # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
    upper_bound = op_magnitude_max or _LEVEL_DENOM
    op_magnitude = max(0., min(op_magnitude, upper_bound))

    # returns a tuple
    level_args = op_level_fn(op_magnitude, op_hparams) if op_level_fn is not None else tuple()
    # convert to np
    level_args_np = np.array(level_args)

    op_compiled = Compiler.compile(op_aug_fn)

    # breakpoint()

    def op(img):
        
        return op_compiled(img, prob=prob, level_args=level_args_np, fillcolor=fillcolor_np, resample=resample_np)

    return op


# we currently don't use this one
_RAND_TRANSFORMS_INCREASING_FFCV = [
    'AutoContrast',
    'Equalize',
    'Invert',
    # 'Rotate',
    # 'PosterizeIncreasing',
    'SolarizeIncreasing',
    # 'SolarizeAdd',
    'ColorIncreasing',
    # 'ContrastIncreasing',
    # 'BrightnessIncreasing',
    # 'SharpnessIncreasing',
    # 'ShearX',
    # 'ShearY',
    # 'TranslateXRel',
    # 'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]

# # we currently don't use this one
# _RAND_TRANSFORMS = [
#     'AutoContrast',
#     'Equalize',
#     'Invert',
#     'Rotate',
#     'Posterize',
#     'Solarize',
#     'SolarizeAdd',
#     'Color',
#     'Contrast',
#     'Brightness',
#     'Sharpness',
#     'ShearX',
#     'ShearY',
#     'TranslateXRel',
#     'TranslateYRel',
#     #'Cutout'  # NOTE I've implement this as random erasing separately
# ]

# ### # we use this one
# _RAND_INCREASING_TRANSFORMS = [
#     'AutoContrast',
#     'Equalize',
#     'Invert',
#     'Rotate',
#     'PosterizeIncreasing',
#     'SolarizeIncreasing',
#     'SolarizeAdd',
#     'ColorIncreasing',
#     'ContrastIncreasing',
#     'BrightnessIncreasing',
#     'SharpnessIncreasing',
#     'ShearX',
#     'ShearY',
#     'TranslateXRel',
#     'TranslateYRel',
#     #'Cutout'  # NOTE I've implement this as random erasing separately
# ]



# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
# all add up to 100
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'Posterize': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


# make a version that finds the numpy versions only
def rand_augment_ops_ffcv(magnitude=10, hparams=None, transforms=None):
    """
    Runs a for loop to call transform factory on each op
    """
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS_INCREASING_FFCV

    return [augment_op_factory(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugmentFFCV(Operation):
    """
    This single custom ffcv function utilizes many non-ffcv functions (compiled).  It 
    allows us to not need to write custom ffcv functions for each transform, but instead
    compiles each non-ffcv function it uses.

    """

    def __init__(self, transforms, num_layers=2, choice_weights=None):
        self.transforms = transforms  # can't have any duplicates, will break!
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def generate_code(self):
        # Compiler.set_enabled(False)  # turn on for debugging only
        parallel_range = Compiler.get_iterator()

        # inner function can't handle any object vars
        num_layers = self.num_layers
        replace = False  # I believe we want this

        # can't use p weights in numba
        # choice_weights = tuple(self.choice_weights) if self.choice_weights is not None else None

        transforms_compiled_list = []

        # compile all possible helper functions/transforms here
        for t in parallel_range(len(self.transforms)):
            trx_comp = Compiler.compile(self.transforms[t])
            transforms_compiled_list.append(trx_comp)

        # must use tuple
        transforms_compiled = tuple(transforms_compiled_list)

        def op(images, dst):
            for i in parallel_range(images.shape[0]):
                im = images[i]
                new_im = im  # must use new var name
                
                # select n_layer of transforms to use
                ixs = np.random.choice(len(transforms_compiled), num_layers, replace=replace)
                
                # indexing tuple while calling transform must be hardcorded, can't use for loop
                # of a variable to index w/numba
                for j in parallel_range(num_layers):
                    t = ixs[j]
                    if t == 0:
                        transforms_compiled[0](new_im)
                    elif t == 1:
                        transforms_compiled[1](new_im)
                    elif t == 2:
                        transforms_compiled[2](new_im)
                    elif t == 3:
                        transforms_compiled[3](new_im)
                    elif t == 4:
                        transforms_compiled[4](new_im)

                dst[i] = new_im

            return dst
        
        op.is_parallel = True
        return op

    def declare_state_and_memory(self, previous_state):
        # Current shape
        h, w, c = previous_state.shape
        
        # Logic here to determine if the op actually changes the shape
        # new_shape = (h // 2, w // 2, c)
        # TODO: 
        pass
        
        new_state = previous_state
        new_shape = (h, w, c)
        return (new_state, AllocationQuery(new_shape, previous_state.dtype))

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for trx in self.transforms:
            fs += f'\n\t{trx}'
        fs += ')'
        return fs


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
        'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS_INCREASING_FFCV  # note the change here
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float('inf')
            hparams.setdefault('magnitude_std', mstd)
        elif key == 'mmax':
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault('magnitude_max', int(val))
        elif key == 'inc':  # we use this one 
            if bool(val):
                # transforms = _RAND_INCREASING_TRANSFORMS
                transforms = _RAND_TRANSFORMS_INCREASING_FFCV  # overide for now
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':  # we don't use
            num_layers = int(val)
        elif key == 'w':  # we don't use, currently breaks with this, not sure why
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'

    # returns a list of object ops (not functions, but uses __call__)
    ra_ops = rand_augment_ops_ffcv(magnitude=magnitude, hparams=hparams, transforms=transforms)

    # ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    # probably will be None
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)

    # eventually replace with RandAugmentFFCV version
    return RandAugmentFFCV(ra_ops, num_layers, choice_weights=choice_weights)


#### end autoaug_cv2.py





def transforms_noaug_train(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(tfl)


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range

    primary_tfl = []

    # add the primary ffcv transforms here

    # primary_tfl = [
    #     RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    # if hflip > 0.:
    #     primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    # if vflip > 0.:
    #     primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)

        if auto_augment.startswith('rand'):  ####   we use this one  ####
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        
        # # we don't use this ones, don't implement
        # elif auto_augment.startswith('augmix'):
        #     aa_params['translate_pct'] = 0.3
        #     secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        # else:
        #     secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    # final_tfl = []
    # if use_prefetcher:
    #     # prefetcher and collate will handle tensor conversion and norm
    #     final_tfl += [ToNumpy()]
    # else:
    #     final_tfl += [
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=torch.tensor(mean),
    #             std=torch.tensor(std))
    #     ]
    #     if re_prob > 0.:
    #         final_tfl.append(
    #             RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    # if separate:
    #     return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    # else:
    #     return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

    return secondary_tfl


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        # if is_training and no_aug:
        #     assert not separate, "Cannot perform split augmentation with no_aug"
        #     transform = transforms_noaug_train(
        #         img_size,
        #         interpolation=interpolation,
        #         use_prefetcher=use_prefetcher,
        #         mean=mean,
        #         std=std)
        # elif is_training:
        transform = transforms_imagenet_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate)
        # else:
        #     assert not separate, "Separate transforms not supported for validation preprocessing"
        #     transform = transforms_imagenet_eval(
        #         img_size,
        #         interpolation=interpolation,
        #         use_prefetcher=use_prefetcher,
        #         mean=mean,
        #         std=std,
        #         crop_pct=crop_pct)

    return transform

    # overide for testing, send back list of Op (ffcv functions)
    # make sure not to use ToTensor, Normalize, or use Compose(), dataset will handle
    # return [RandAugmentFFCV([auto_contrast, auto_contrast], num_layers=2)]