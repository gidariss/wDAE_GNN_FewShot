from __future__ import print_function

import json
import os
import os.path

import h5py
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance

import low_shot_learning.utils as utils

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = '/datasets_local/ImageNet/'
_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH = './data/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json'
_MEAN_PIXEL = [0.485, 0.456, 0.406]
_STD_PIXEL = [0.229, 0.224, 0.225]


class ImageJitter:
    def __init__(self, transformdict):
        transformtypedict=dict(
            Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color
        )
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


class ImageNet(data.Dataset):
    def __init__(
        self,
        split='train',
        use_geometric_aug=True,
        use_simple_geometric_aug=False,
        use_color_aug=True):
        # use_geometric_aug: If True geometric augmentations are used for the
        # images of the training split.
        # use_color_aug: if True color augmentations are used for the images
        # of the test/val split.

        self.split = split
        assert split in ('train', 'val')
        self.name = 'ImageNet_Split_' + split

        data_dir = _IMAGENET_DATASET_DIR
        print('==> Loading ImageNet dataset - split {0}'.format(split))
        print('==> ImageNet directory: {0}'.format(data_dir))

        transform_train = []
        assert not (use_simple_geometric_aug and use_geometric_aug)
        if use_geometric_aug:
            transform_train.append(transforms.RandomResizedCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        elif use_simple_geometric_aug:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.RandomCrop(224))
            transform_train.append(transforms.RandomHorizontalFlip())
        else:
            transform_train.append(transforms.Resize(256))
            transform_train.append(transforms.CenterCrop(224))

        if use_color_aug:
            jitter_params = {'Brightness': 0.4, 'Contrast': 0.4, 'Color': 0.4}
            transform_train.append(ImageJitter(jitter_params))

        transform_train.append(lambda x: np.asarray(x))
        transform_train.append(transforms.ToTensor())
        transform_train.append(
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL))

        self.trainsform_train = transform_train

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])

        self.transform = transform_train if split=='train' else transform_test
        print('==> transform: {0}'.format(self.transform))
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if split=='train' else val_dir
        self.data = datasets.ImageFolder(split_dir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

    #@profile
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNetLowShot(ImageNet):
    def __init__(
        self,
        phase='train',
        split='train',
        do_not_use_random_transf=False):

        assert phase in ('train', 'test', 'val')
        assert split in ('train', 'val')

        use_aug = (phase=='train') and (do_not_use_random_transf==False)

        ImageNet.__init__(
            self, split=split, use_geometric_aug=use_aug, use_color_aug=use_aug)

        self.phase = phase
        self.split = split
        self.name = 'ImageNetLowShot_Phase_' + phase + '_Split_' + split
        print('==> Loading ImageNet dataset (for few-shot benchmark) - phase {0}'.
            format(phase))

        #***********************************************************************
        with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
            label_idx = json.load(f)
        base_classes = label_idx['base_classes']
        novel_classes_val_phase = label_idx['novel_classes_1']
        novel_classes_test_phase = label_idx['novel_classes_2']
        #***********************************************************************

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats==1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val_phase if (self.phase=='val') else
                novel_classes_test_phase)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0


class ImageNetLowShotFeatures:
    def __init__(
        self,
        data_dir, # path to the directory with the saved ImageNet features.
        image_split='train', # the image split of the ImageNet that will be loaded.
        phase='train', # whether the dataset will be used for training, validating, or testing a model.
        ):
        assert image_split in ('train', 'val')
        assert phase in ('train', 'val', 'test')

        self.phase = phase
        self.image_split = image_split
        self.name = (f'ImageNetLowShotFeatures_ImageSplit_{self.image_split}'
                     f'_Phase_{self.phase}')

        dataset_file = os.path.join(
            data_dir, 'ImageNet_' + self.image_split + '.h5')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        #***********************************************************************
        with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
            label_idx = json.load(f)
        base_classes = label_idx['base_classes']
        base_classes_val_split = label_idx['base_classes_1']
        base_classes_test_split = label_idx['base_classes_2']
        novel_classes_val_split = label_idx['novel_classes_1']
        novel_classes_test_split = label_idx['novel_classes_2']
        #***********************************************************************

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats==1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)

        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val_split if (self.phase=='val') else
                novel_classes_test_split)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
            self.base_classes_eval_split = (
                base_classes_val_split if (self.phase=='val') else
                base_classes_test_split)
            self.base_classes_subset = self.base_classes_eval_split


    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)


class ImageNetFeatures:
    def __init__(
        self,
        data_dir, # path to the directory with the saved ImageNet features.
        split='train', # the image split of the ImageNet that will be loaded.
        ):
        assert split in ('train', 'val')

        self.split = split
        self.name = (f'ImageNetFeatures_ImageSplit_{self.split}')

        dataset_file = os.path.join(
            data_dir, 'ImageNet_' + self.split + '.h5')
        self.data_file = h5py.File(dataset_file, 'r')
        self.count = self.data_file['count'][0]
        self.features = self.data_file['all_features'][...]
        self.labels = self.data_file['all_labels'][:self.count].tolist()

        self.label2ind = utils.buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return int(self.count)
