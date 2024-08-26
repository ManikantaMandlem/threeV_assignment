# taken from pytorch segmentation reference transforms
# source: https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py

import random
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ReSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size, interpolation=Image.BILINEAR)
        target = F.resize(target, self.size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomAffine:
    def __init__(self, affine_prob, max_angle=30, max_translate=0.1, max_shear=10):
        self.affine_prob = affine_prob
        self.max_angle = max_angle
        self.max_translate = max_translate
        self.max_shear = max_shear

    def __call__(self, image, target):
        if random.random() < self.affine_prob:
            # get all the necessary parameters for affine transf.
            angle = np.random.uniform(10, self.max_angle)
            htranslate = np.random.uniform(0.05, self.max_translate)
            vtranslate = np.random.uniform(0.05, self.max_translate)
            shear = np.random.uniform(5, self.max_shear)
            # peform the affine on image and mask
            image = F.affine(
                img=image,
                angle=angle,
                translate=[htranslate, vtranslate],
                scale=1,
                shear=shear,
                interpolation=Image.BILINEAR,
                fill=0.498,
            )
            target = F.affine(
                img=target,
                angle=angle,
                translate=[htranslate, vtranslate],
                scale=1,
                shear=shear,
                interpolation=Image.NEAREST,
                fill=0,
            )
        return image, target


class GaussianBlur:
    def __init__(self, blur_prob):
        self.blur_prob = blur_prob
        self.blur_transform = transforms.Compose(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))]
        )

    def __call__(self, image, target):
        if random.random() < self.blur_prob:
            image = self.blur_transform(image)
        return image, target
