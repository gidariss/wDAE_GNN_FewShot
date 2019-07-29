import torch
import torch.nn.functional as F

import low_shot_learning.architectures.classifiers.utils as cutils
from low_shot_learning.architectures.classifiers.cosine_classifier_with_weight_generator import \
    CosineClassifierWithWeightGeneration
from low_shot_learning.architectures.classifiers.weights_denoising_autoencoder import WeightsDAE


def reconstruction_loss(outputs, targets):
    # Both outputs and targets have shape:
    # [batch_size x num_nodes x num_features]
    assert outputs.dim() == 3
    assert targets.dim() == 3
    assert outputs.size() == targets.size()

    # Since we use cosine classifier the weights must be L_2 normalized.
    targets = F.normalize(targets, p=2, dim=targets.dim()-1, eps=1e-12)
    outputs = F.normalize(outputs, p=2, dim=outputs.dim()-1, eps=1e-12)
    # return the L2 squared loss (averaged over the first 2 dimensions, i.e.,
    # batch_size and num_nodes).
    return (targets - outputs).pow(2).mean() * outputs.size(2)


class CosineClassifierWithDAEWeightGeneration(CosineClassifierWithWeightGeneration):
    def __init__(
        self,
        dae_config,
        num_features,
        num_classes,
        global_pooling,
        scale_cls=10.0,
        learn_scale=True):

        super(CosineClassifierWithDAEWeightGeneration, self).__init__(
            num_features, num_classes, global_pooling, scale_cls, learn_scale)

        self.targets_as_input = (
            dae_config['targets_as_input']
            if ('targets_as_input' in dae_config) else False)

        self.comp_reconstruction_loss = (
            dae_config['comp_reconstruction_loss']
            if ('comp_reconstruction_loss' in dae_config) else True)

        self.weights_dae_generator = WeightsDAE(dae_config)

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
        self.num_base = num_base

        #***********************************************************************
        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        num_novel = labels_train.size(2)

        if features_train.dim() == 5:
            features_train = cutils.preprocess_5D_features(
                features_train, self.global_pooling)
        assert features_train.dim() == 3
        assert features_train.size(2) == self.num_features
        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)

        #***********************************************************************
        #******* Generate classification weights for base & novel classes ******
        weight_base = weight_base.detach()

        if ((self.targets_as_input or self.comp_reconstruction_loss) and
            self.training):
            novel_ids = self._novel_ids
            assert novel_ids.size(1) == num_novel
            weight_novel_target = self.weight_base[novel_ids.view(-1)].detach()
            weight_novel_target = weight_novel_target.view(
                meta_batch_size, num_novel, self.num_features)

        if self.targets_as_input and self.training:
            weight_novel = weight_novel_target
        else:
            # Estimate the initial classification weights for the novel classes
            # by computing the average of the feature vectors of their training
            # examples.
            weight_novel = cutils.average_train_features(
                features_train, labels_train)

        input_weights = torch.cat([weight_base, weight_novel], dim=1)
        # Since we use cosine classifier the weights must be L_2 normalized.
        input_weights = F.normalize(input_weights, p=2, dim=2, eps=1e-12)

        output_weights = self.weights_dae_generator(input_weights)
        #***********************************************************************

        if self.training and self.comp_reconstruction_loss:
            targets_weights = torch.cat([weight_base, weight_novel_target], 1)
            self.reconstruction_loss = reconstruction_loss(
                output_weights, targets_weights)
        else:
            self.reconstruction_loss = None

        return output_weights


def create_model(opt):
    return CosineClassifierWithDAEWeightGeneration(
        dae_config=opt['dae_config'],
        num_features=opt['num_features'],
        num_classes=opt['num_classes'],
        global_pooling=opt['global_pooling'],
        scale_cls=opt['scale_cls'],
        learn_scale=(opt['learn_scale'] if ('learn_scale' in opt) else True))
