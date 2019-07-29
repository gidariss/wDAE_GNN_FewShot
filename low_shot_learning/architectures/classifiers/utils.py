import numpy as np
import torch
import torch.nn as nn

import low_shot_learning.architectures.tools as tools


class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=20.0,
        learn_scale=False,
        bias=False):
        super(CosineClassifier, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0/num_channels))
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return tools.cosine_fully_connected_layer(
            x_in, self.weight.t(), scale=self.scale_cls, bias=self.bias)

    def extra_repr(self):
        s = (
            'num_channels={0}, num_classes={1}, scale_cls={2} (learnable={3})'
            .format(self.num_channels, self.num_classes, self.scale_cls.item(),
            self.scale_cls.requires_grad))

        if self.bias is None:
            s += ', bias=False'
        return s


def average_train_features(features_train, labels_train):
    labels_train_transposed = labels_train.transpose(1,2)
    weight_novel = torch.bmm(labels_train_transposed, features_train)
    weight_novel = weight_novel.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(
            weight_novel))

    return weight_novel


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        return average_train_features(features_train, labels_train)


def preprocess_5D_features(features, global_pooling):
    meta_batch_size, num_examples, channels, height, width = features.size()
    features = features.view(
        meta_batch_size * num_examples, channels, height, width)

    if global_pooling:
        features = tools.global_pooling(features, pool_type='avg')

    features = features.view(meta_batch_size, num_examples, -1)

    return features
