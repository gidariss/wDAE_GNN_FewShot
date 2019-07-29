import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import low_shot_learning.architectures.classifiers.utils as cutils
import low_shot_learning.architectures.tools as tools


class CosineClassifierWithWeightGeneration(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        global_pooling,
        scale_cls=10.0,
        learn_scale=True,
        bias_novel=None):
        super(CosineClassifierWithWeightGeneration, self).__init__()


        self.global_pooling = global_pooling
        self.num_features = num_features
        self.num_classes = num_classes
        weight_base = torch.FloatTensor(
            self.num_classes, self.num_features).normal_(
                0.0, np.sqrt(2.0/self.num_features))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)

        self.learn_scale = learn_scale
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls),
            requires_grad=self.learn_scale)

        self.bias_novel = bias_novel


    def get_classification_weights(
            self, base_ids, features_train=None, labels_train=None):
        """Gets the classification weights of the base and novel categories.

        This routine returns the classification weight of the base and novel
        classes. The latter are returned only if the input arguments
        features_train and labels_train are not None.

        Args:
            base_ids: A 2D tensor with shape [meta_batch_size x num_base] that
                for each training episode in the the batch it includes the
                indices of the base categories that are being used.
                `meta_batch_size` is the number of training episodes in the
                batch and `num_base` is the number of base classes.
            features_train: A 3D tensor with shape
                [meta_batch_size x num_train_examples x num_channels] that
                represents the `num_channels`-dimensional feature vectors of the
                training examples of each training episode in the batch.
                `num_train_examples` is the number of train examples in each
                training episode. Those training examples are from the novel
                classes.
            labels_train: A 3D tensor with shape
                [meta_batch_size x num_train_examples x num_novel] that
                represents the labels (encoded as 1-hot vectors of lenght
                num_novel) of the training examples of each training episode in
                the batch. `num_novel` is the number of novel classes.

        Returns:
            classification_weights: A 3D tensor of shape
                [meta_batch_size x num_classes x num_channels]
                that includes the `num_channels`-dimensional classification
                weight vectors of the classes involved on each training episode
                in the batch. If the training data for the novel classes are not
                provided (i.e., features_train or labels_train are None) then
                classification_weights includes only the classification
                weights of the base classes; in this case num_channels is equal
                to `num_base`. Otherwise, classification_weights includes the
                classification weight vectors of both base and novel classses;
                in this case `num_classes` is equal to `num_base` + `num_novel`.
        """

        #***********************************************************************
        #******** Get the classification weights for the base categories *******
        meta_batch_size, num_base = base_ids.size()
        weight_base = self.weight_base[base_ids.view(-1)]
        weight_base = weight_base.view(meta_batch_size, num_base, -1)

        #***********************************************************************
        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        if features_train.dim() == 5:
            features_train = cutils.preprocess_5D_features(
                features_train, self.global_pooling)

        assert(features_train.dim() == 3)
        assert(features_train.size(2) == self.num_features)

        #***********************************************************************
        #******* Generate classification weights for the novel categories ******
        features_train = F.normalize(
            features_train, p=2, dim=features_train.dim()-1, eps=1e-12)

        num_novel = labels_train.size(2)
        weight_novel = cutils.average_train_features(
            features_train, labels_train)
        weight_novel = weight_novel.view(
            meta_batch_size, num_novel, self.num_features)
        #***********************************************************************

        # Concatenate the base and novel classification weights and return them.
        weight_both = torch.cat([weight_base, weight_novel], dim=1)
        # weight_both shape:
        # [meta_batch_size x (num_base + num_novel) x self.num_classes]
        self.num_novel = num_novel
        self.num_base = num_base

        return weight_both


    def apply_classification_weights(self, features, classification_weights):
        """Applies the classification weight vectors to the feature vectors.

        Args:
            features: A 3D tensor of shape
                [meta_batch_size x num_test_examples x num_channels] with the
                feature vectors (of `num_channels` length) of each example on
                each trainining episode in the batch. `meta_batch_size` is the
                number of training episodes in the batch and `num_test_examples`
                is the number of test examples of each training episode.
            classification_weights: A 3D tensor of shape
                [meta_batch_size x num_classes x num_channels]
                that includes the `num_channels`-dimensional classification
                weight vectors of the `num_classes` classes used on
                each training episode in the batch. `num_classes` is the number
                of classes (e.g., the number of base classes plus the number
                of novel classes) used on each training episode.

        Return:
            classification_scores: A 3D tensor with shape
                [meta_batch_size x num_test_examples x nK] that represents the
                classification scores of the test examples for the `nK`
                categories.
        """
        if features.dim() == 5:
            features = cutils.preprocess_5D_features(
                features, self.global_pooling)
        assert(features.dim() == 3)
        assert(features.size(2) == self.num_features)

        classification_scores = tools.batch_cosine_fully_connected_layer(
            features, classification_weights.transpose(1,2),
            scale=self.scale_cls)

        if self.bias_novel is not None:
            classification_scores_base = (
                classification_scores[:, :, :self.num_base])
            classification_scores_novel = (
                self.bias_novel * classification_scores[:, :, self.num_base:])
            classification_scores = torch.cat(
                [classification_scores_base, classification_scores_novel],
                dim=2)

        self._cls_weights = classification_weights

        return classification_scores

    def parse_base_ids(self, base_ids, meta_batch_size=1):
        if base_ids is None:
            num_base = self.weight_base.size(0)
            device = 'cuda' if self.weight_base.is_cuda else 'cpu'
            base_ids = (torch.LongTensor(
                list(range(num_base))).view(1, -1).repeat(
                meta_batch_size, 1))
            base_ids = base_ids.to(device)

        return base_ids

    def forward(
        self,
        features_test,
        base_ids=None,
        features_train=None,
        labels_train=None):
        """Recognize on the test examples both base and novel categories.

        Recognize on the test examples (i.e., `features_test`) both base and
        novel categories using the approach proposed on our CVPR2018 paper
        "Dynamic Few-Shot Visual Learning without Forgetting". In order to
        classify the test examples the provided training data for the novel
        categories (i.e., `features_train` and `labels_train`) are used in order
        to generate classification weight vectors of those novel categories and
        then those classification weight vectors are applied on the features of
        the test examples.

        Args:
            features_test: A 3D tensor with shape
                [meta_batch_size x num_test_examples x num_channels] that
                represents the `num_channels`-dimensional feature vectors of the
                test examples each training episode in the batch. Those examples
                can be both from base and novel classes. `meta_batch_size` is
                the number of training episodes in the batch,
                `num_test_examples` is the number of test examples in each
                training episode, and `num_channels` is the number of feature
                channels.
            base_ids: A 2D tensor with shape [meta_batch_size x num_base] that
                for each training episode in the batch it includes the indices
                of the base classes that are being used. `num_base` is the
                number of base classes.
            features_train: A 3D tensor with shape
                [meta_batch_size x num_train_examples x num_channels] that
                represents the `num_channels`-dimensional feature vectors of the
                training examples of each training episode in the batch.
                `num_train_examples` is the number of train examples in each
                training episode. Those training examples are from the novel
                classes. If features_train is None then the current function
                will only return the classification scores for the base classes.
            labels_train: A 3D tensor with shape
                [meta_batch_size x num_train_examples x num_novel] that
                represents the labels (encoded as 1-hot vectors of lenght
                `num_novel`) of the training examples of each training episode
                in the batch. `num_novel` is the number of novel classes. If
                labels_train is None then the current function will return only
                the classification scores for the base classes.

        Return:
            classsification_scores: A 3D tensor with shape
                [meta_batch_size x num_test_examples x (num_base + num_novel)]
                that represents the classification scores of the test examples
                for the num_base and num_novel novel classes. If features_train
                or labels_train are None then only the classification scores of
                the base classes are returned. In that case the shape of
                cls_scores is [meta_batch_size x num_test_examples x num_base].
        """
        base_ids = self.parse_base_ids(base_ids, features_test.size(0))
        classsification_weights = self.get_classification_weights(
            base_ids, features_train, labels_train)
        classsification_scores = self.apply_classification_weights(
            features_test, classsification_weights)
        return classsification_scores


def create_model(opt):
    return CosineClassifierWithWeightGeneration(
        num_features=opt['num_features'],
        num_classes=opt['num_classes'],
        global_pooling=opt['global_pooling'],
        scale_cls=opt['scale_cls'],
        learn_scale=(opt['learn_scale'] if ('learn_scale' in opt) else True),
        bias_novel=(opt['bias_novel'] if ('bias_novel' in opt) else None))
