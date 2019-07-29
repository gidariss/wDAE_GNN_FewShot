from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

import low_shot_learning.algorithms.classification.utils as cls_utils
import low_shot_learning.utils as utils


def few_shot_feature_classification(
    classifier, features_test, features_train, labels_train_1hot, labels_test,
    base_ids=None):
    if base_ids is not None:
        classification_scores = classifier(
            features_test=features_test,
            features_train=features_train,
            labels_train=labels_train_1hot,
            base_ids=base_ids)
    else:
        classification_scores = classifier(
            features_test=features_test,
            features_train=features_train,
            labels_train=labels_train_1hot)

    assert(classification_scores.dim() == 3)

    classification_scores = classification_scores.view(
        classification_scores.size(0) * classification_scores.size(1), -1)
    labels_test = labels_test.view(-1)
    assert(classification_scores.size(0) == labels_test.size(0))

    loss = F.cross_entropy(classification_scores, labels_test)

    return classification_scores, loss


def compute_accuracy_metrics(scores, labels, num_base, record={}, string_id=''):
    assert(isinstance(record, dict))

    if string_id != '':
        string_id = '_' + string_id

    if labels.dim() > 1:
        labels = labels.view(scores.size(0))

    if num_base > 0:
        record['AccuracyBoth' + string_id] = utils.top1accuracy(scores, labels)
        # scores = scores.cpu()
        # labels = labels.cpu()

        base_indices = torch.nonzero(labels < num_base).view(-1)
        novel_indices = torch.nonzero(labels >= num_base).view(-1)
        if base_indices.dim() != 0 and base_indices.size(0) > 0:
            scores_base = scores[base_indices][:, :num_base]
            labels_base = labels[base_indices]
            record['AccuracyBase' + string_id] = utils.top1accuracy(
                scores_base, labels_base)

        scores_novel = scores[novel_indices,:][:, num_base:]
        labels_novel = labels[novel_indices] - num_base
        record['AccuracyNovel' + string_id] = utils.top1accuracy(
            scores_novel, labels_novel)
    else:
        record['AccuracyNovel' + string_id] = utils.top1accuracy(scores, labels)

    return record


def fewshot_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images_train,
    labels_train,
    labels_train_1hot,
    images_test,
    labels_test,
    is_train,
    base_ids=None,
    feature_name=None,
    classification_coef=1.0,
    reconstruction_coef=0.0):

    assert(images_train.dim() == 5)
    assert(images_test.dim() == 5)
    assert(images_train.size(0) == images_test.size(0))
    assert(images_train.size(2) == images_test.size(2))
    assert(images_train.size(3) == images_test.size(3))
    assert(images_train.size(4) == images_test.size(4))
    assert(labels_train.dim() == 2)
    assert(labels_test.dim() == 2)
    assert(labels_train.size(0) == labels_test.size(0))
    assert(labels_train.size(0) == images_train.size(0))

    if (feature_name and
        isinstance(feature_name, (list, tuple)) and
        len(feature_name) > 1):
        assert is_train is False
        assert reconstruction_coef == 0.0
        assert classification_coef == 1.0
        return fewshot_classification_multiple_features(
            feature_extractor=feature_extractor,
            feature_extractor_optimizer=feature_extractor_optimizer,
            classifier=classifier,
            classifier_optimizer=classifier_optimizer,
            images_train=images_train,
            labels_train=labels_train,
            labels_train_1hot=labels_train_1hot,
            images_test=images_test,
            labels_test=labels_test,
            is_train=is_train,
            base_ids=base_ids,
            feature_name=feature_name)

    meta_batch_size = images_train.size(0)

    if is_train: # zero the gradients
        if feature_extractor_optimizer:
            feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        batch_size_train = images_train.size(0)
        # batch_size_test = images_test.size(0)
        images = torch.cat([images_train, images_test], dim=0)

    train_feature_extractor = (
        is_train and (feature_extractor_optimizer is not None))
    with torch.set_grad_enabled(train_feature_extractor):
        # Extract features from the train and test images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name)

    if not train_feature_extractor:
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        features = features.detach()

    with torch.set_grad_enabled(is_train):
        features_train = features[:batch_size_train]
        features_test = features[batch_size_train:]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)

        classification_scores, loss = few_shot_feature_classification(
            classifier, features_test, features_train, labels_train_1hot,
            labels_test, base_ids)
        record['loss'] = loss.item()
        loss_total = loss * classification_coef

        if is_train and (reconstruction_coef > 0.0):
            rec_loss = classifier.reconstruction_loss
            assert(rec_loss is not None)
            loss_total = loss_total + reconstruction_coef * rec_loss
            record['rec_loss'] = rec_loss.item()
            record['tot_loss'] = loss_total.item()
        #*******************************************************************

    with torch.no_grad():
        num_base = base_ids.size(1) if (base_ids is not None) else 0
        record = compute_accuracy_metrics(
            classification_scores, labels_test, num_base, record)

    if is_train:
        loss_total.backward()
        if feature_extractor_optimizer:
            feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record


def fewshot_classification_multiple_features(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images_train,
    labels_train,
    labels_train_1hot,
    images_test,
    labels_test,
    is_train,
    feature_name,
    base_ids=None):

    assert is_train is False
    assert feature_name and isinstance(feature_name, (list, tuple))

    meta_batch_size = images_train.size(0)
    num_base = base_ids.size(1) if (base_ids is not None) else 0

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        batch_size_train = images_train.size(0)
        images = torch.cat([images_train, images_test], dim=0)

    with torch.set_grad_enabled(is_train):
        # Extract features from the train and test images.
        features = cls_utils.extract_features(
            feature_extractor, images, feature_name=feature_name)
        assert len(features) == len(feature_name)

        for i, feature_name_i in enumerate(feature_name):
            features_train = features[i][:batch_size_train]
            features_test = features[i][batch_size_train:]
            features_train = utils.add_dimension(
                features_train, meta_batch_size)
            features_test = utils.add_dimension(
                features_test, meta_batch_size)

            if isinstance(classifier, (list, tuple)):
                assert len(classifier) == len(feature_name)
                classifier_this = classifier[i]
            else:
                classifier_this = classifier

            classification_scores, loss = few_shot_feature_classification(
                classifier_this, features_test, features_train,
                labels_train_1hot, labels_test, base_ids)
            record['loss_'+feature_name_i] = loss.item()

            with torch.no_grad():
                record = compute_accuracy_metrics(
                    classification_scores, labels_test, num_base, record,
                    string_id=feature_name_i)

    return record


def compute_95confidence_intervals(
    record,
    episode,
    num_episodes,
    store_accuracies,
    metrics=['AccuracyNovel',]):

    if episode==0:
        store_accuracies = {metric: [] for metric in metrics}

    for metric in metrics:
        store_accuracies[metric].append(record[metric])
        if episode == (num_episodes - 1):
            # Compute std and confidence interval of the 'metric' accuracies.
            accuracies = np.array(store_accuracies[metric])
            stds = np.std(accuracies, 0)
            record[metric + '_std'] = stds
            record[metric + '_cnf'] = 1.96*stds/np.sqrt(num_episodes)

    return record, store_accuracies


def compute_weight_orthogonality_loss(cls_weights):

    nKall = cls_weights.size(1)
    device = 'cuda' if cls_weights.is_cuda else 'cpu'
    orthogonality_loss = torch.add(
        torch.bmm(cls_weights, cls_weights.transpose(1,2)),
        -torch.eye(nKall).to(device).view(1, nKall, nKall)).abs().mean()

    return orthogonality_loss
