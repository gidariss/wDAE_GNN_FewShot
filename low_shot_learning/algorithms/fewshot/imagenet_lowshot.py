from __future__ import print_function

import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

import low_shot_learning.algorithms.fewshot.fewshot as fewshot
import low_shot_learning.utils as utils
import low_shot_learning.architectures.tools as tools


def compute_top1_and_top5_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def softmax_with_novel_prior(scores, novel_inds, base_inds, prior_m):
    scores = torch.exp(scores)
    scores_novel = scores[:, novel_inds]
    scores_base = scores[:, base_inds]
    tol = 0.0000001
    scores_novel *= (
        prior_m /
        (tol + torch.sum(scores_novel, dim=1, keepdim=True).expand_as(scores_novel)))
    scores_base *= (
        (1.0 - prior_m) /
        (tol + torch.sum(scores_base, dim=1, keepdim=True).expand_as(scores_base)))
    scores[:, novel_inds] = scores_novel
    scores[:, base_inds] = scores_base
    return scores


class ImageNetLowShot(fewshot.FewShot):
    def __init__(self, opt):
        super().__init__(opt)
        self.keep_best_model_metric_name = 'top5_novel'

    def preprocess_novel_training_data(self, training_data):
        """Preprocess the novel training data."""

        images_train, labels_train, Kids, num_base, num_novel = training_data
        self.num_base = num_base
        self.num_novel = num_novel

        # Insert an extra singleton dimension.
        images_train = images_train.unsqueeze(dim=0)
        labels_train = labels_train.unsqueeze(dim=0)
        Kids = Kids.unsqueeze(dim=0)

        self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
        self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
        self.tensors['Kids'].resize_(Kids.size()).copy_(Kids)
        labels_train = self.tensors['labels_train']

        labels_train_1hot_size = list(labels_train.size()) + [num_novel,]
        dim = len(labels_train_1hot_size) - 1
        labels_train = labels_train.unsqueeze(dim=labels_train.dim())
        self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
            dim,labels_train - num_base, 1)

    def add_novel_categories(self, nove_cat_training_data):
        """Add the training data of the novel categories to the model."""

        feature_extractor = self.networks['feature_extractor']
        classifier = self.networks['classifier']
        feature_extractor.eval()
        classifier.eval()

        self.preprocess_novel_training_data(nove_cat_training_data)

        images = self.tensors['images_train'].detach()
        labels_train_1hot = self.tensors['labels_train_1hot'].detach()
        Kids = self.tensors['Kids'].detach()
        base_ids = None if (self.num_base==0) else Kids[:,:self.num_base].contiguous()

        with torch.no_grad():
            #*******************************************************************
            #****************** EXTRACT FEATS FROM EXEMPLARS *******************
            meta_batch_size = images.size(0)
            images = utils.convert_from_5d_to_4d(images)
            features_train = feature_extractor(images)
            features_train = utils.add_dimension(features_train, meta_batch_size)

            #*******************************************************************
            #****************** GET CLASSIFICATION WEIGHTS *********************
            # The following routine returns the classification weight vectors of
            # both the base and then novel categories. For the novel categories,
            # the classification weight vectors are generated using the training
            # features for those novel cateogories.
            clsWeights = classifier.get_classification_weights(
                base_ids=base_ids,
                features_train=features_train,
                labels_train=labels_train_1hot)
            #*******************************************************************

        self.tensors['clsWeights'] = clsWeights.clone().detach()

    def evaluate_model_on_test_images(
        self, data_loader, base_classes, novel_classes, exp_id='', prior_m=0.8):
        """Evaluate the model.

        It is assumed that the user has already called the routine
        add_novel_categories() before calling this function.

        Args:
            data_loader: data loader that feeds test images and lables in order
                to evaluatethe model.
            base_classes: A list with the labels of the base categories that
                will be used for evaluation.
            novel_classes: A list with the labels of the novel categories that
                will be used for evaluation.
            exp_id: A string with the id of the experiment.
            prior_m: A scalar in the range [0, 1.0] that represents the prior
                for whether a test image comes from the novel or base classes.
        """

        feature_extractor = self.networks['feature_extractor']
        classifier = self.networks['classifier']
        feature_extractor.eval()
        classifier.eval()
        clsWeights = self.tensors['clsWeights']

        both_classes = base_classes + novel_classes
        # Not valid classes are those that do not belong neighter to the base
        # nor the nor the novel classes.
        nKall = self.num_base + self.num_novel
        not_valid_classes = list(set(range(nKall)).difference(set(both_classes)))
        not_valid_classes_torch = torch.LongTensor(not_valid_classes).to(self.device)
        base_classes_torch = torch.LongTensor(base_classes).to(self.device)
        novel_classes_torch = torch.LongTensor(novel_classes).to(self.device)

        top1, top1_novel, top1_base, top1_prior = None, None, None, None
        top5, top5_novel, top5_base, top5_prior = None, None, None, None
        all_labels = None
        for idx, batch in enumerate(tqdm(data_loader(0))):
            images_test, labels_test = batch
            self.tensors['images_test'].resize_(
                images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(
                labels_test.size()).copy_(labels_test)
            images_test = self.tensors['images_test'].detach()
            labels_test = self.tensors['labels_test'].detach()
            num_test_examples = images_test.size(0)

            with torch.no_grad():
                features = feature_extractor(images_test)
                features = features.view(1, num_test_examples, -1)
                scores = classifier.apply_classification_weights(
                    features, clsWeights)
                scores = scores.view(num_test_examples, -1)

                scores_prior = softmax_with_novel_prior(
                    scores.clone(), novel_classes_torch, base_classes_torch, prior_m)

                scores[:, not_valid_classes_torch] = -1000
                top1_this, top5_this = compute_top1_and_top5_accuracy(
                    scores, labels_test)
                top1 = (
                    top1_this if top1 is None
                    else np.concatenate((top1, top1_this)))
                top5 = (
                    top5_this if top5 is None
                    else np.concatenate((top5, top5_this)))

                scores_prior[:, not_valid_classes_torch] = -1000
                top1_this, top5_this = compute_top1_and_top5_accuracy(
                    scores_prior, labels_test)
                top1_prior = (
                    top1_this if top1_prior is None
                    else np.concatenate((top1_prior, top1_this)))
                top5_prior = (
                    top5_this if top5_prior is None
                    else np.concatenate((top5_prior, top5_this)))

                scores_novel = scores.clone()
                scores_novel[:, base_classes_torch] = -1000
                top1_this, top5_this = compute_top1_and_top5_accuracy(scores_novel, labels_test)
                top1_novel = top1_this if top1_novel is None else np.concatenate((top1_novel, top1_this))
                top5_novel = top5_this if top5_novel is None else np.concatenate((top5_novel, top5_this))

                scores_base = scores.clone()
                scores_base[:, novel_classes_torch] = -1000
                top1_this, top5_this = compute_top1_and_top5_accuracy(scores_base, labels_test)
                top1_base = top1_this if top1_base is None else np.concatenate((top1_base, top1_this))
                top5_base = top5_this if top5_base is None else np.concatenate((top5_base, top5_this))

                labels_test_np = labels_test.cpu().numpy()
                all_labels = labels_test_np if all_labels is None else np.concatenate((all_labels, labels_test_np))

        is_novel = np.in1d(all_labels, np.array(novel_classes))
        is_base = np.in1d(all_labels, np.array(base_classes))
        is_either = is_novel | is_base

        top1_novel = 100*np.mean(top1_novel[is_novel])
        top1_novel_all = 100*np.mean(top1[is_novel])
        top1_base = 100*np.mean(top1_base[is_base])
        top1_base_all = 100*np.mean(top1[is_base])
        top1_all = 100*np.mean(top1[is_either])
        top1_all_prior = 100*np.mean(top1_prior[is_either])

        top5_novel = 100*np.mean(top5_novel[is_novel])
        top5_novel_all = 100*np.mean(top5[is_novel])
        top5_base = 100*np.mean(top5_base[is_base])
        top5_base_all = 100*np.mean(top5[is_base])
        top5_all = 100*np.mean(top5[is_either])
        top5_all_prior = 100*np.mean(top5_prior[is_either])

        self.logger.info('Experiment {0}'.format(exp_id))
        self.logger.info(
            '==> Top 5 Accuracies: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top5_novel, top5_base, top5_all, top5_novel_all, top5_base_all, top5_all_prior))
        self.logger.info(
            '==> Top 1 Accuracies: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top1_novel, top1_base, top1_all, top1_novel_all, top1_base_all, top1_all_prior))

        results_array = np.array(
            [top5_novel, top5_base, top5_all, top5_novel_all, top5_base_all, top5_all_prior,
             top1_novel, top1_base, top1_all, top1_novel_all, top1_base_all, top1_all_prior]).reshape(1,-1)

        return results_array

    def lowshot_avg_results(self, results_all, exp_id=''):
        results_all = np.concatenate(results_all, axis=0)
        num_eval_experiments = results_all.shape[0]

        mu_results = results_all.mean(axis=0)
        top5_novel = mu_results[0]
        top5_base = mu_results[1]
        top5_all = mu_results[2]
        top5_novel_all = mu_results[3]
        top5_base_all = mu_results[4]
        top5_all_prior = mu_results[5]

        top1_novel = mu_results[6]
        top1_base = mu_results[7]
        top1_all = mu_results[8]
        top1_novel_all = mu_results[9]
        top1_base_all = mu_results[10]
        top1_all_prior = mu_results[11]

        std_results  = results_all.std(axis=0)
        ci95_results = 1.96*std_results/np.sqrt(results_all.shape[0])

        top5_novel_ci95 = ci95_results[0]
        top5_base_ci95 = ci95_results[1]
        top5_all_ci95 = ci95_results[2]
        top5_novel_all_ci95 = ci95_results[3]
        top5_base_all_ci95 = ci95_results[4]
        top5_all_prior_ci95 = ci95_results[5]

        top1_novel_ci95 = ci95_results[6]
        top1_base_ci95 = ci95_results[7]
        top1_all_ci95 = ci95_results[8]
        top1_novel_all_ci95 = ci95_results[9]
        top1_base_all_ci95 = ci95_results[10]
        top1_all_prior_ci95 = ci95_results[11]


        self.logger.info('----------------------------------------------------------------')
        self.logger.info('Average results of {0} experiments: {1}'.format(
            num_eval_experiments, exp_id))
        self.logger.info(
            '==> Top 5 Accuracies:      [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top5_novel, top5_base, top5_all, top5_novel_all, top5_base_all, top5_all_prior))
        self.logger.info(
            '==> Top 5 conf. intervals: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top5_novel_ci95, top5_base_ci95, top5_all_ci95, top5_novel_all_ci95, top5_base_all_ci95, top5_all_prior_ci95))
        self.logger.info('----------------------------------------------------------------')
        self.logger.info(
            '==> Top 1 Accuracies:      [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top1_novel, top1_base, top1_all, top1_novel_all, top1_base_all, top1_all_prior))
        self.logger.info(
            '==> Top 1 conf. intervals: [Novel: {0:3.2f} | Base: {1:3.2f} | All {2:3.2f} | Novel vs All {3:3.2f} | Base vs All {4:3.2f} | All prior {5:3.2f}]'
            .format(top1_novel_ci95, top1_base_ci95, top1_all_ci95, top1_novel_all_ci95, top1_base_all_ci95, top1_all_prior_ci95))
        self.logger.info('----------------------------------------------------------------')

        results = {}
        results['top5_novel'] = round(top5_novel, 2)
        results['top5_base'] = round(top5_base, 2)
        results['top5_all'] = round(top5_all, 2)
        results['top5_novel_all'] = round(top5_novel_all, 2)
        results['top5_base_all'] = round(top5_base_all, 2)
        results['top5_all_prior'] = round(top5_all_prior, 2)

        results['top5_novel_ci95'] = round(top5_novel_ci95, 2)
        results['top5_base_ci95'] =round( top5_base_ci95, 2)
        results['top5_all_ci95'] = round(top5_all_ci95, 2)
        results['top5_novel_all_ci95'] = round(top5_novel_all_ci95, 2)
        results['top5_base_all_ci95'] = round(top5_base_all_ci95, 2)
        results['top5_all_prior_ci95'] = round(top5_all_prior_ci95, 2)

        return results

    def evaluate(
        self, dloader, num_eval_exp=20, prior=0.8, suffix=''):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        self.logger.info('Num exemplars: %d' % dloader.nExemplars)
        self.logger.info('Num evaluation experiments: %d' % num_eval_exp)
        self.logger.info('Prior: %f' % prior)

        results = []
        # Run args_opt.num_exp different number of evaluation experiments (each time
        # sampling a different set of training images for the the novel categories).
        for exp_id in range(num_eval_exp):
            # Sample training data for the novel categories from the training set of
            # ImageNet.
            nove_cat_data = dloader.sample_training_data_for_novel_categories(
                exp_id=exp_id)
            # Feed the training data of the novel categories to the algorithm.
            self.add_novel_categories(nove_cat_data)
            # Evaluate on the validation images of ImageNet.
            results_this = self.evaluate_model_on_test_images(
                data_loader=dloader,
                base_classes=dloader.base_category_label_indices(),
                novel_classes=dloader.novel_category_label_indices(),
                exp_id='Exp_id = ' + str(exp_id),
                prior_m=prior)
            results.append(results_this)

        # Print the average results.
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        avg_results = self.lowshot_avg_results(results, exp_id='')

        eval_stats = utils.DAverageMeter('eval', self._run)
        eval_stats.update(avg_results)
        eval_stats.log()
        self.add_stats_to_tensorboard_writer(
            eval_stats.average(), 'test_')

        return eval_stats
