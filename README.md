# ffcv_transform_factory_randaug

A working repo for implementing RandAugment w/ImageNet in the FFCV library for speedup.

Will eventually have these transforms (those marked "done" have been tested):

_RAND_TRANSFORMS_INCREASING_FFCV = [
    'AutoContrast',  # done
    'Equalize',  # done
    'Invert',  # done
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing', # done
    'SolarizeAdd',
    'ColorIncreasing',  # done
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    'Cutout'  # maybe implement as random erasing separately
]



Increasing just means the level can be adjusted for that function.

Each transform just needs to be in pure numpy AND numba supported, which is the JIT compiler for python. That makes
it more restrictive, and sometimes requiring more complex numpy functions.

If you're able to contribute any of the other functions, please let me know!  I can drop them in and get RandAugment
fully functional!

TODOs:

- add primary and "final" transforms, right now the create_transform function only returns RandAugment object, and my
dataloader handles adding the primary/final transforms from FFCV, easy fix later.
- other transforms