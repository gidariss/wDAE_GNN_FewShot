from __future__ import print_function

import numpy as np
import torchnet as tnt


def generate_element_list(list_size, dataset_size):
    if list_size == dataset_size:
        return list(range(dataset_size))
    elif list_size < dataset_size:
        return np.random.choice(
            dataset_size, list_size, replace=False).tolist()
    else: # list_size > list_size
        num_times = list_size // dataset_size
        residual = list_size % dataset_size
        assert((num_times * dataset_size + residual) == list_size)
        elem_list = list(range(dataset_size)) * num_times
        if residual:
            elem_list += np.random.choice(
                dataset_size, residual, replace=False).tolist()

        return elem_list


class SimpleDataloader:
    def __init__(
        self, dataset, batch_size, train, num_workers=4, epoch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_size = len(dataset)
        self.epoch_size = epoch_size if epoch_size else self.dataset_size
        self.train = train

    def get_iterator(self, epoch=0):
        def load_fun_(idx):
            #idx0 = idx
            img, label = self.dataset[idx % len(self.dataset)]
            #print('idx0={0}, len(d)={1}, idx={2}, label={3}'.format(
            #    idx0, len(self.dataset), idx % len(self.dataset), label))
            return img, label

        elem_list = generate_element_list(self.epoch_size, self.dataset_size)

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=elem_list, load=load_fun_)

        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train,
            drop_last=self.train)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator()

    def __len__(self):
        return self.epoch_size // self.batch_size
