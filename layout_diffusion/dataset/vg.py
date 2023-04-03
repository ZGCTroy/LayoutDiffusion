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
from collections import defaultdict
import torch
import torchvision.transforms as T
import numpy as np
import h5py
import PIL

from layout_diffusion.dataset.util import image_normalize
from layout_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from torch.utils.data import Dataset
import json
from PIL import Image


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 max_objects_per_image=10, max_num_samples=None, mask_size=32,
                 use_orphaned_objects=True,
                 left_right_flip=False, min_object_size=32, use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=[]
                 ):
        super(VgSceneGraphDataset, self).__init__()

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.image_size = image_size
        self.min_object_size = min_object_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects_per_image = max_objects_per_image
        self.max_num_samples = max_num_samples

        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0
        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    # self.image_
                    self.image_paths = [str(path, encoding="utf-8") for path in list(v)]
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

        # get specific image ids or get specific number of images
        selected_idx = []
        self.specific_image_ids = specific_image_ids
        if self.specific_image_ids:
            specific_image_ids_set = set(specific_image_ids)
            for idx, image_path in enumerate(self.image_paths):
                if image_path in specific_image_ids_set:
                    selected_idx.append(idx)
                    specific_image_ids_set.remove(image_path)
                if len(specific_image_ids_set) == 0:
                    break

            if len(specific_image_ids_set) > 0:
                for image_path in list(specific_image_ids_set):
                    print('image path: {} is not found'.format(image_path))

            assert len(specific_image_ids_set) == 0
        elif self.max_num_samples:
            selected_idx = [idx for idx in range(self.max_num_samples)]

        if selected_idx:
            print('selected_idx = {}'.format(selected_idx))
            self.image_paths = [self.image_paths[idx] for idx in selected_idx]
            for k in list(self.data.keys()):
                self.data[k] = self.data[k][selected_idx]

    def check_with_relation(self, image_index):
        '''
        :param obj_idxs: the idxs of objects of image
        :return: with_relations = [True, False, ....], shape=(O,), O is the number of objects
        '''
        obj_idxs = range(self.data['objects_per_image'][image_index].item())
        with_relations = [False for i in obj_idxs]
        for r_idx in range(self.data['relationships_per_image'][image_index]):
            s = self.data['relationship_subjects'][image_index, r_idx].item()
            o = self.data['relationship_objects'][image_index, r_idx].item()
            with_relations[s] = True
            with_relations[o] = True
        without_relations = [not i for i in with_relations]
        return with_relations, without_relations

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def get_init_meta_data(self, image_index):
        layout_length = self.max_objects_per_image + 2
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__']),
            'is_valid_obj': torch.zeros([layout_length]),
            'filename': self.image_paths[image_index].replace('/', '_').split('.')[0]
        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        meta_data['obj_class'][0] = self.vocab['object_name_to_idx']['__image__']
        meta_data['is_valid_obj'][0] = 1.0

        return meta_data

    def load_image(self, index):
        with open(os.path.join(self.image_dir, self.image_paths[index]), 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGB')
        return image

    def __len__(self):
        num = self.data['object_names'].size(0)
        assert num == len(self.image_paths)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        """

        # Figure out which objects appear in relationships and which don't
        with_relations, without_relations = self.check_with_relation(image_index=index)  # (O,)

        image = self.load_image(index)
        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        image = np.array(image, dtype=np.float32) / 255.0

        H, W, _ = image.shape
        num_obj = self.data['objects_per_image'][index].item()
        obj_bbox = self.data['object_boxes'][index].numpy()[:num_obj]
        obj_class = self.data['object_names'][index].numpy()[:num_obj]
        is_valid_obj = (obj_class >= 0)

        # get meta data
        meta_data = self.get_init_meta_data(image_index=index)
        meta_data['width'], meta_data['height'] = W, H

        meta_data['with_relations'] = with_relations

        # filter invalid bbox
        obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)

        # flip
        if self.left_right_flip:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)

        # random crop image and its bbox
        if self.use_MinIoURandomCrop:
            image, updated_obj_bbox, updated_obj_class, tmp_is_valid_obj = self.MinIoURandomCrop(image, obj_bbox[is_valid_obj], obj_class[is_valid_obj])

            tmp_idx = 0
            tmp_tmp_idx = 0
            for idx, is_valid in enumerate(is_valid_obj):
                if is_valid:
                    if tmp_is_valid_obj[tmp_idx]:
                        obj_bbox[idx] = updated_obj_bbox[tmp_tmp_idx]
                        tmp_tmp_idx += 1
                    else:
                        is_valid_obj[idx] = False
                    tmp_idx += 1

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape

        obj_bbox_with_relations = torch.FloatTensor(obj_bbox[is_valid_obj & with_relations])
        obj_bbox_without_relations = torch.FloatTensor(obj_bbox[is_valid_obj & without_relations])
        obj_class_with_relations = torch.LongTensor(obj_class[is_valid_obj & with_relations])
        obj_class_without_relations = torch.LongTensor(obj_class[is_valid_obj & without_relations])

        obj_bbox_with_relations[:, 0::2] = obj_bbox_with_relations[:, 0::2] / W
        obj_bbox_with_relations[:, 1::2] = obj_bbox_with_relations[:, 1::2] / H
        obj_bbox_without_relations[:, 0::2] = obj_bbox_without_relations[:, 0::2] / W
        obj_bbox_without_relations[:, 1::2] = obj_bbox_without_relations[:, 1::2] / H

        num_selected = min(obj_bbox_with_relations.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox_with_relations.shape[0]), num_selected)
        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox_with_relations[selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = obj_class_with_relations[selected_obj_idxs]
        meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] = num_selected
        meta_data['num_add'] = 0

        if num_selected < self.max_objects_per_image and self.use_orphaned_objects:
            num_add = min(self.max_objects_per_image - num_selected, obj_bbox_without_relations.shape[0])
            if num_add > 0:
                selected_obj_idxs = random.sample(range(obj_bbox_without_relations.shape[0]), num_add)
                meta_data['obj_bbox'][1 + num_selected:1 + num_selected + num_add] = obj_bbox_without_relations[selected_obj_idxs]
                meta_data['obj_class'][1 + num_selected:1 + num_selected + num_add] = obj_class_without_relations[selected_obj_idxs]
                meta_data['is_valid_obj'][1 + num_selected:1 + num_selected + num_add] = 1.0
                meta_data['num_add'] = num_add

        meta_data['obj_class_name'] = [self.vocab['object_idx_to_name'][int(class_id)] for class_id in meta_data['obj_class']]
        meta_data['num_obj'] = meta_data['num_selected'] + meta_data['num_add'] - 1

        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)

        return self.transform(image), meta_data


def vg_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_meta_data = defaultdict(list)
    all_imgs = []

    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        if key in ['obj_bbox', 'obj_class', 'is_valid_obj'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_vg_dsets(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']

    params = cfg.data.parameters
    with open(os.path.join(params.root_dir, params.vocab_json), 'r') as f:
        vocab = json.load(f)

    # print(vocab)
    vocab['object_name_to_idx']['__image__'] = 0
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
        use_orphaned_objects=params.use_orphaned_objects,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids
    )

    num_imgs = len(dataset)
    print('%s dataset has %d images' % (mode, num_imgs))

    return dataset
