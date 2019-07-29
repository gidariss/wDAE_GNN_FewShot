from __future__ import print_function

import os
import os.path
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import low_shot_learning.utils as utils

# Set the appropriate paths of the datasets here.
_MINI_IMAGENET_DATASET = '/datasets_local/MiniImagenet/'

_MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
_STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]


class MiniImageNetBase(data.Dataset):
    def __init__(
        self,
        transform_test,
        transform_train,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False):

        data_dir = _MINI_IMAGENET_DATASET
        print('==> Download MiniImageNet dataset at {0}'.format(data_dir))
        file_train_categories_train_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        file_train_categories_val_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_val.pickle')
        file_train_categories_test_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_train_phase_test.pickle')
        file_val_categories_val_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            data_dir, 'miniImageNet_category_split_test.pickle')

        self.phase = phase
        if load_single_file_split:
            assert file_split in ('category_split_train_phase_train',
                                  'category_split_train_phase_val',
                                  'category_split_train_phase_test',
                                  'category_split_val',
                                  'category_split_test')
            self.file_split = file_split
            self.name = 'MiniImageNet_' + file_split

            print(
                '==> Loading mini ImageNet dataset - phase {0}'
                .format(file_split))

            file_to_load = os.path.join(
                data_dir,
                'miniImageNet_{0}.pickle'.format(file_split))

            data = utils.load_pickle_data(file_to_load)
            self.data = data['data']
            self.labels = data['labels']
            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
        else:
            assert(phase in ('train', 'val', 'test', 'trainval'))
            self.name = 'MiniImageNet_' + phase

            print('Loading mini ImageNet dataset - phase {0}'.format(phase))
            if self.phase=='train':
                # Loads the training classes (and their data) as base classes
                data_train = utils.load_pickle_data(
                    file_train_categories_train_phase)
                self.data = data_train['data']
                self.labels = data_train['labels']

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase=='trainval':
                # Loads the training + validation classes (and their data) as
                # base classes
                data_train = utils.load_pickle_data(
                    file_train_categories_train_phase)
                data_val = utils.load_pickle_data(
                    file_val_categories_val_phase)
                self.data = np.concatenate(
                    [data_train['data'], data_val['data']], axis=0)
                self.labels = data_train['labels'] + data_val['labels']

                self.label2ind = utils.buildLabelIndex(
                    self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)
                self.labelIds_base = self.labelIds
                self.num_cats_base = len(self.labelIds_base)

            elif self.phase=='val' or self.phase=='test':
                # Uses the validation / test classes (and their data) as novel
                # as novel class data and the vaditation / test image split of
                # the training classes for the base classes.

                if self.phase=='test':
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base classes.
                    data_base = utils.load_pickle_data(
                        file_train_categories_test_phase)
                    # load data that will be use for evaluating the few-shot
                    # recogniton accuracy on the novel classes.
                    data_novel = utils.load_pickle_data(
                        file_test_categories_test_phase)
                else: # phase=='val'
                    # load data that will be used for evaluating the recognition
                    # accuracy of the base classes.
                    data_base = utils.load_pickle_data(
                        file_train_categories_val_phase)
                    # load data that will be use for evaluating the few-shot
                    # recogniton accuracy on the novel classes.
                    data_novel = utils.load_pickle_data(
                        file_val_categories_val_phase)

                self.data = np.concatenate(
                    [data_base['data'], data_novel['data']], axis=0)
                self.labels = data_base['labels'] + data_novel['labels']

                self.label2ind = utils.buildLabelIndex(self.labels)
                self.labelIds = sorted(self.label2ind.keys())
                self.num_cats = len(self.labelIds)

                self.labelIds_base = utils.buildLabelIndex(data_base['labels']).keys()
                self.labelIds_novel = utils.buildLabelIndex(data_novel['labels']).keys()
                self.num_cats_base = len(self.labelIds_base)
                self.num_cats_novel = len(self.labelIds_novel)
                intersection = set(self.labelIds_base) & set(self.labelIds_novel)
                assert len(intersection) == 0
            else:
                raise ValueError('Not valid phase {0}'.format(self.phase))

        self.transform_test = transform_test
        self.transform_train = transform_train
        if ((self.phase=='test' or self.phase=='val') or
            (do_not_use_random_transf==True)):
            self.transform = self.transform_test
        else:
            self.transform = self.transform_train

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


class MiniImageNet(MiniImageNetBase):
    def __init__(
        self,
        phase='train',
        image_size=84,
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False):

        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)

        if image_size==84:

            transform_test = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize])
        else:
            assert image_size > 0

            transform_test = transforms.Compose([
                transforms.Resize(image_size),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])

            transform_train = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize,])


        MiniImageNetBase.__init__(
            self,
            transform_test=transform_test,
            transform_train=transform_train,
            phase=phase,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)


class MiniImageNet80x80(MiniImageNet):
    def __init__(
        self,
        phase='train',
        load_single_file_split=False,
        file_split=None,
        do_not_use_random_transf=False):

        MiniImageNet.__init__(
            self,
            phase=phase,
            image_size=80,
            load_single_file_split=load_single_file_split,
            file_split=file_split,
            do_not_use_random_transf=do_not_use_random_transf)


def load_features_labels(dataset_file):
    data_file = h5py.File(dataset_file, 'r')
    count = data_file['count'][0]
    features = data_file['all_features'][...]
    labels = data_file['all_labels'][:count].tolist()
    features = features[:count,:]

    return data_file, count, features, labels


class MiniImageNetFeatures(data.Dataset):
    def __init__(self, data_directory, phase='train'):
        file_train_categories_train_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_train.json')
        file_train_categories_val_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_val.json')
        file_train_categories_test_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_train_phase_test.json')
        file_val_categories_val_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_val.json')
        file_test_categories_test_phase = os.path.join(
            data_directory,
            'MiniImageNet_category_split_test.json')

        self.phase = phase
        assert phase in ('train', 'val', 'test', 'trainval')
        self.name = 'MiniImageNetFeatures_Phase_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            _, _, features, labels = load_features_labels(
                file_train_categories_train_phase)

            self.features = features
            self.labels = labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='trainval':
            _, _, features_train, labels_train = load_features_labels(
                file_train_categories_train_phase)
            _, _, features_val, labels_val = load_features_labels(
                file_val_categories_val_phase)

            self.features = np.concatenate(
                [features_train, features_val], axis=0)
            self.labels = labels_train + labels_val

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                _, _, base_features, base_labels = load_features_labels(
                    file_train_categories_test_phase)

                # load data that will be use for evaluating the few-shot
                # recogniton accuracy on the novel categories.
                _, _, novel_features, novel_labels = load_features_labels(
                    file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the
                # recognition accuracy of the base categories.
                _, _, base_features, base_labels = load_features_labels(
                    file_train_categories_val_phase)

                # load data that will be use for evaluating the few-shot
                # recogniton accuracy on the novel categories.
                _, _, novel_features, novel_labels = load_features_labels(
                    file_val_categories_val_phase)

            self.features = np.concatenate(
                [base_features, novel_features], axis=0)
            self.labels = base_labels + novel_labels

            self.label2ind = utils.buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = utils.buildLabelIndex(base_labels).keys()
            self.labelIds_novel = utils.buildLabelIndex(novel_labels).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

    def __getitem__(self, index):
        features_this = torch.Tensor(self.features[index]).view(-1,1,1)
        label_this = self.labels[index]
        return features_this, label_this

    def __len__(self):
        return len(self.data)
