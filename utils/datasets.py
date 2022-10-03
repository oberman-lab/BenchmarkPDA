import os
import numpy as np
import numbers
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import utils.data_list
from PIL import Image, ImageOps

def transform_train(resize_size: int = 256, crop_size: int = 224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def transform_train_augmented(resize_size: int = 256, crop_size: int = 224, jitter_param: int = 0.4):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def transform_test(resize_size: int = 256, crop_size: int = 224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



# class ResizeImage():
#     def __init__(self, size):
#         if isinstance(size, int):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#     def __call__(self, img):
#         th, tw = self.size
#         return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image"""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


# def image_test(resize_size=256, crop_size=224):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#     #ten crops for image when validation, input the data_transforms dictionary
#     start_first = 0
#     start_center = (resize_size - crop_size - 1) / 2
#     start_last = resize_size - crop_size - 1

#     return transforms.Compose([
#         ResizeImage(resize_size),
#         PlaceCrop(crop_size, start_center, start_center),
#         transforms.ToTensor(),
#         normalize
#         ])

def transforms_10crop(resize_size: int = 256, crop_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = {}
    data_transforms[0] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        ForceFlip(),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[1] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        ForceFlip(),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[2] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        ForceFlip(),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[3] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        ForceFlip(),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[4] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        ForceFlip(),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[5] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[6] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[7] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[8] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms[9] = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])
    return data_transforms