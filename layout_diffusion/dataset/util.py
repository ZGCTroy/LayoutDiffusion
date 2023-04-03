#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import PIL
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import crop, resize
import numpy as np

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# IMAGENET_MEAN = [0., 0., 0.]
# IMAGENET_STD = [1.0, 1.0, 1.0]

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def blank(x):
    return x


def image_normalize():
    # return blank
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def image_unnormalize(rescale_image=False):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def image_unnormalize_batch(imgs, rescale=False):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) or (C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) or (C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    # if isinstance(imgs, torch.autograd.Variable):
    #   imgs = imgs.data
    # imgs = imgs.cpu().clone()
    deprocess_fn = image_unnormalize(rescale_image=rescale)
    imgs_de = []
    if imgs.dim() == 4:
        for i in range(imgs.size(0)):
            img_de = deprocess_fn(imgs[i])[None]
            # img_de = img_de.mul(255).clamp(0, 255).byte()
            # img_de = img_de.mul(255).clamp(0, 255)
            imgs_de.append(img_de)
        imgs_de = torch.cat(imgs_de, dim=0)
        return imgs_de
    elif imgs.dim() == 3:
        img_de = deprocess_fn(imgs)
        return img_de
    else:
        raise NotImplementedError


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)



def get_cropped_image(obj_bboxes, images, image_size=256, cropped_size=32, antialias=True):
    '''

    :param obj_bboxes: # N * L * 4, (x0, y0, w, h)
    :param images:    # N * 3 * H * W
    :param cropped_size: mask_size
    :return:
    '''

    rounded_obj_bbox = obj_bboxes.clone()
    height, width = images.shape[2], images.shape[3]
    rounded_obj_bbox[:,:, 0::2] = rounded_obj_bbox[:,:,0::2] * width
    rounded_obj_bbox[:,:,1::2] = rounded_obj_bbox[:,:,1::2] * height
    rounded_obj_bbox = torch.round(rounded_obj_bbox)
    rounded_obj_bbox = rounded_obj_bbox.long()
    # rounded_obj_bbox[:, :, 2] = torch.where(
    #     rounded_obj_bbox[:, :, 2] >= 1,
    #     rounded_obj_bbox[:, :, 2],
    #     1
    # )
    # rounded_obj_bbox[:, :, 3] = torch.where(
    #     rounded_obj_bbox[:, :, 3] >= 1,
    #     rounded_obj_bbox[:, :, 3],
    #     1
    # )
    bs, length = rounded_obj_bbox.shape[0], rounded_obj_bbox.shape[1]

    cropped_images = []
    device = obj_bboxes.device
    for image_id in range(rounded_obj_bbox.shape[0]):
        for object_id, object in enumerate(rounded_obj_bbox[image_id]):
            if torch.equal(obj_bboxes[image_id][object_id], torch.zeros((4,), device=device)):
                cropped_images.append(torch.zeros((3, cropped_size, cropped_size), device=device))
                continue

            x0, y0, x1, y1 = object

            cropped_image = crop(images[image_id], top=y0, left=x0, height=max(y1 - y0, 1), width=max(x1 - x0, 1))

            if antialias:
                cropped_image = resize(cropped_image, size=[cropped_size, cropped_size], antialias=True)
            else:
                cropped_image = resize(cropped_image, size=[cropped_size, cropped_size])

            cropped_images.append(cropped_image)

    cropped_images = torch.stack(cropped_images).reshape(bs, length, 3, cropped_size, cropped_size)

    return cropped_images


