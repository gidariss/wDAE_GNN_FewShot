import torch
import torch.nn as nn
import torch.nn.functional as F

import low_shot_learning.architectures.classifiers.utils as cutils
import low_shot_learning.architectures.tools as tools


class FewShotClassifierWithPrototypes(nn.Module):
    def __init__(self, global_pooling, scale_cls=10.0, learn_scale=True):
        super(FewShotClassifierWithPrototypes, self).__init__()

        self.global_pooling = global_pooling
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale)

    def forward(self, features_test, features_train, labels_train):

        #******* Generate classification weights for the novel categories ******
        if features_train.dim() == 5:
            features_train = cutils.preprocess_5D_features(
                features_train, self.global_pooling)
        assert(features_train.dim() == 3)

        meta_batch_size = features_train.size(0)
        num_novel = labels_train.size(2)
        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        classification_weights = cutils.average_train_features(
            features_train, labels_train)
        classification_weights = classification_weights.view(
            meta_batch_size, num_novel, -1)
        #***********************************************************************

        if features_test.dim() == 5:
            features_test = cutils.preprocess_5D_features(
                features_test, self.global_pooling)
        assert(features_test.dim() == 3)

        classification_scores = tools.batch_cosine_fully_connected_layer(
            features_test, classification_weights.transpose(1,2),
            scale=self.scale_cls)

        return classification_scores


def create_model(opt):
    return FewShotClassifierWithPrototypes(
        global_pooling=opt['global_pooling'],
        scale_cls=opt['scale_cls'],
        learn_scale=opt['learn_scale'])
