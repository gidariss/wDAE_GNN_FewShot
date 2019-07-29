from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np

import low_shot_learning.utils as utils


def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float).mean() * 100, top5_correct.astype(float).mean() * 100


def extract_features(feature_extractor, images, feature_name=None):
    if feature_name:
        if isinstance(feature_name, str):
            feature_name = [feature_name,]
        assert isinstance(feature_name, (list, tuple))

        features = feature_extractor(images, out_feat_keys=feature_name)
    else:
        features = feature_extractor(images)

    return features


def classification_task(classifier, features, labels, base_ids=None):
    if base_ids is not None:
        assert(base_ids.dim() == 2)
        batch_size = features.size(0)
        meta_batch_size = base_ids.size(0)
        features = utils.add_dimension(features, dim_size=meta_batch_size)
        scores = classifier(features_test=features, base_ids=base_ids)
        scores = scores.view(batch_size, -1)
    else:
        scores = classifier(features)

    loss = F.cross_entropy(scores, labels)

    return scores, loss


def object_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    base_ids=None,
    feature_name=None):

    if isinstance(feature_name, (list, tuple)) and len(feature_name) > 1:
        assert base_ids is None
        return object_classification_multiple_features(
            feature_extractor=feature_extractor,
            feature_extractor_optimizer=feature_extractor_optimizer,
            classifier=classifier,
            classifier_optimizer=classifier_optimizer,
            images=images,
            labels=labels,
            is_train=is_train,
            feature_name=feature_name)

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train: # Zero gradients.
        if feature_extractor_optimizer:
            feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    train_feature_extractor = (
        is_train and (feature_extractor_optimizer is not None))
    with torch.set_grad_enabled(train_feature_extractor):
        # Extract features from the images.
        features = extract_features(
            feature_extractor, images, feature_name=feature_name)

    if not train_feature_extractor:
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        features = features.detach()

    with torch.set_grad_enabled(is_train):
        # Perform the object classification task.
        scores_classification, loss_classsification = classification_task(
            classifier, features, labels, base_ids)
        loss_total = loss_classsification
        record['loss'] = loss_total.item()

    with torch.no_grad(): # Compute accuracies.
        AccuracyTop1, AccuracyTop5 = compute_top1_and_top5_accuracy(
            scores_classification, labels)
        record['AccuracyTop1'] = AccuracyTop1
        record['AccuracyTop5'] = AccuracyTop5
        #record['Accuracy'] = utils.top1accuracy(scores_classification, labels)

    if is_train: # Backward loss and apply gradient steps.
        loss_total.backward()
        if feature_extractor_optimizer:
            feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record


def object_classification_multiple_features(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    feature_name):

    assert isinstance(feature_name, (list, tuple)) and len(feature_name) > 1
    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train: # Zero gradients.
        if feature_extractor_optimizer:
            feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    train_feature_extractor = (
        is_train and (feature_extractor_optimizer is not None))
    with torch.set_grad_enabled(train_feature_extractor):
        # Extract features from the images.
        features = extract_features(
            feature_extractor, images, feature_name=feature_name)
        assert len(features) == len(feature_name)

    if not train_feature_extractor:
        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        for i in range(len(features)):
            features[i] = features[i].detach()

    with torch.set_grad_enabled(is_train):
        # Perform the object classification task.
        scores = classifier(features)
        assert len(scores) == len(feature_name)

        losses = []
        for i in range(len(scores)):
            losses.append(F.cross_entropy(scores[i], labels))
            record['loss_' + feature_name[i]] = losses[i].item()

            with torch.no_grad(): # Compute accuracies.
                AccuracyTop1, AccuracyTop5 = compute_top1_and_top5_accuracy(
                    scores[i], labels)
                record['AccuracyTop1_' + feature_name[i]] = AccuracyTop1
                record['AccuracyTop5_' + feature_name[i]] = AccuracyTop5

        loss_total = torch.stack(losses).sum()

    if is_train: # Backward loss and apply gradient steps.
        loss_total.backward()
        if feature_extractor_optimizer:
            feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record
