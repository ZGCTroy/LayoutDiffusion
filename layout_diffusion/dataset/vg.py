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

import os
import random

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import resized_crop
from skimage.transform import resize as imresize
import numpy as np
import h5py
import PIL

from layout_diffusion.dataset.util import Resize
from torch.utils.data import Dataset
import json


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 max_objects_per_image=10, max_num_samples=None, mask_size=32,
                 use_orphaned_objects=True,
                 left_right_flip=False):
        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects_per_image = max_objects_per_image
        self.max_num_samples = max_num_samples
        self.left_right_flip = left_right_flip

        self.origin_transform = T.Compose([T.ToTensor()])
        self.transform = T.Compose([Resize(image_size), T.ToTensor()])

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = [str(path, encoding="utf-8") for path in list(v)]
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

        # for index in range(self.data['object_names'].size(0)):
        #     num_obj = len(self.data['object_boxes'][index])
        #     for obj in self.data['object_boxes'][index]:
        #         obj_list = obj.tolist()
        #         x, y, w, h = obj_list
        # assert 0<=x<WW, print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x+w, y+h, WW, HH))
        # assert 0<=x+w<WW, print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x+w, y+h, WW, HH))
        # assert 0<=y<HH, print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x+w, y+h, WW, HH))
        # assert 0<=y+h<HH, print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x+w, y+h, WW, HH))

    def __len__(self):
        num = self.data['object_names'].size(0)
        assert num == len(self.image_paths)
        if self.max_num_samples is not None:
            return min(self.max_num_samples, num)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        """

        img_path = os.path.join(self.image_dir, self.image_paths[index])
        flip = False
        if self.left_right_flip and np.random.random() < 0.5:
            flip = True
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as origin_image:
                if flip:
                    origin_image = PIL.ImageOps.mirror(origin_image)
                WW, HH = origin_image.size
                origin_image = origin_image.convert('RGB')
                image = self.transform(origin_image)
                origin_image = self.origin_transform(origin_image)

        H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects_per_image - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects_per_image)
        if len(obj_idxs) < self.max_objects_per_image - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects_per_image - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        # The first object will be the special __image__ object
        layout_length = self.max_objects_per_image + 2
        objs = torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__'])
        objs[0] = self.vocab['object_name_to_idx']['__image__']
        boxes = torch.zeros([layout_length, 4])
        boxes[0] = torch.FloatTensor([0, 0, 1, 1])
        cropped_obj_images = torch.zeros([layout_length, 3, self.mask_size, self.mask_size])
        cropped_obj_images[0] = torch.FloatTensor(imresize(origin_image, (3, self.mask_size, self.mask_size), mode='constant'))
        is_valid_obj = torch.zeros([layout_length])
        is_valid_obj[0] = 1.0

        # max_objects_per_image
        for i, obj_idx in enumerate(obj_idxs):
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()

            if (WW - x) < 32 or (HH - y) < 32:
                print('Invalid BBox: x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x + w, y + h, WW, HH))
                continue

            # x0 = np.clip(x0, 0, WW-1)
            # y0 = np.clip(y0, 0, HH-1)
            # x1 = np.clip(x1, 0, WW-x)
            # y1 = np.clip(y1, 0, HH-y)
            x = np.clip(x, 0, WW - 1)
            y = np.clip(y, 0, HH - 1)
            w = np.clip(w, 0, WW - x)
            h = np.clip(h, 0, HH - y)

            # if not (0 <= x < WW and 0 <= y < HH and 0 <= (x + w) <= WW and 0 <= (y + h) <= HH):
            #     print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x + w, y + h, WW, HH))
            #     objs[i + 1] = self.vocab['object_name_to_idx']['__null__']
            #     boxes[i + 1] = torch.FloatTensor([0, 0, 0, 0])
            #     cropped_obj_images[i+1] = torch.zeros([3, self.mask_size, self.mask_size])
            #     continue

            # print('x0 = {}, y0={}, x1={}, y1={}, WW={}, HH={}'.format(x, y, x+w, y+h, WW, HH))
            #

            if flip:
                x = WW - (x + w)

            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(w) / WW
            y1 = float(h) / HH

            is_valid_obj[i + 1] = 1.0
            objs[i + 1] = self.data['object_names'][index, obj_idx].item()
            boxes[i + 1] = torch.FloatTensor([x0, y0, x1, y1])
            cropped_obj_image = origin_image[:, y:y+max(1,h), x:x+max(1,w)]
            cropped_obj_image = imresize(cropped_obj_image, (3, self.mask_size, self.mask_size), mode='constant')
            cropped_obj_image = torch.FloatTensor(cropped_obj_image)
            # cropped_obj_image = resized_crop(img=origin_image, top=y, left=x, height=h, width=w, size=[self.mask_size, self.mask_size])
            cropped_obj_images[i + 1] = cropped_obj_image

        filename = self.image_paths[index]
        filename = filename.replace('/','_')

        return image, objs, boxes, cropped_obj_images, is_valid_obj, filename


def vg_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_imgs, all_padded_objs, all_padded_boxes, all_padded_cropped_obj_images = [], [], [], []
    all_is_valid_objs, all_filenames = [], []

    for i, (img, padded_objs, padded_boxes, padded_cropped_obj_images, is_valid_objs, filename) in enumerate(batch):
        all_imgs.append(img[None])
        all_padded_objs.append(padded_objs)
        all_padded_boxes.append(padded_boxes)
        all_padded_cropped_obj_images.append(padded_cropped_obj_images)
        all_is_valid_objs.append(is_valid_objs)
        all_filenames.append(filename)

    all_imgs = torch.cat(all_imgs)
    all_padded_objs = torch.stack(all_padded_objs)
    all_padded_boxes = torch.stack(all_padded_boxes)
    all_padded_cropped_obj_images = torch.stack(all_padded_cropped_obj_images)
    all_is_valid_objs = torch.stack(all_is_valid_objs)

    out_dict = {
        'obj_class': all_padded_objs,
        'obj_box': all_padded_boxes,
        'obj_image': all_padded_cropped_obj_images,  # (N, L, 3, H, W),
        'is_valid_obj': all_is_valid_objs,  # (N, L)
        'filename': all_filenames
    }

    return all_imgs, out_dict


def build_vg_dsets(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']

    params = cfg.data.parameters
    with open(os.path.join(params.root_dir, params.vocab_json), 'r') as f:
        vocab = json.load(f)

    # print(vocab)
    vocab['object_name_to_idx']['__null__'] = 179
    vocab['object_idx_to_name'].append('__null__')

    dataset = VgSceneGraphDataset(
        vocab=vocab,
        h5_path=os.path.join(params.root_dir, params[mode].h5_path),
        image_dir=os.path.join(params.root_dir, params.image_dir),
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        max_num_samples=params[mode].max_num_samples,
        max_objects_per_image=params.max_objects_per_image,
        left_right_flip=params[mode].left_right_flip,
        use_orphaned_objects=params.use_orphaned_objects
    )

    num_imgs = len(dataset)
    print('%s dataset has %d images' % (mode, num_imgs))

    return dataset
