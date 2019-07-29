from __future__ import print_function

import torch

import low_shot_learning.algorithms.algorithm as algorithm
import low_shot_learning.algorithms.fewshot.utils as fs_utils
import low_shot_learning.algorithms.classification.utils as utils


class Classification(algorithm.Algorithm):
    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        feature_name = opt['feature_name'] if ('feature_name' in opt) else None

        if feature_name:
            assert isinstance(feature_name, (list, tuple))

        self.feature_name = feature_name

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def set_tensors(self, batch):
        assert len(batch) == 2
        images, labels = batch
        self.tensors['images'].resize_(images.size()).copy_(images)
        self.tensors['labels'].resize_(labels.size()).copy_(labels)

        return 'classification'

    def train_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=True)

    def evaluation_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=False)

    def process_batch_classification_task(self, batch, is_train):
        self.set_tensors(batch)

        if is_train and (self.optimizers.get('feature_extractor') is None):
            self.networks['feature_extractor'].eval()

        record = utils.object_classification(
            feature_extractor=self.networks['feature_extractor'],
            feature_extractor_optimizer=self.optimizers.get('feature_extractor'),
            classifier=self.networks['classifier'],
            classifier_optimizer=self.optimizers.get('classifier'),
            images=self.tensors['images'],
            labels=self.tensors['labels'],
            is_train=is_train,
            base_ids=None,
            feature_name=self.feature_name)

        return record
