"""Define a generic class for training and testing learning algorithms."""
from __future__ import print_function

import datetime
import glob
import logging
import os
import os.path
import pathlib

import tensorboardX
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm

import low_shot_learning.architectures as architectures
import low_shot_learning.utils as utils



class Algorithm:
    def __init__(self, opt, _run=None, _log=None):
        self.set_experiment_dir(pathlib.Path(opt['exp_dir']))
        self.set_log_file_handler(_log)

        self.logger.info(f'Algorithm options {opt}')
        self.opt = opt
        self._run = _run
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}
        self.device = 'cpu'

        self.keep_best_model_metric_name = (
            opt['best_metric'] if ('best_metric' in opt) else None)

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        os.makedirs(self.exp_dir, exist_ok=True)

        self.vis_dir = directory_path / 'visuals'
        os.makedirs(self.vis_dir, exist_ok=True)

        self.preds_dir = directory_path / 'preds'
        os.makedirs(self.preds_dir, exist_ok=True)

    def set_log_file_handler(self, _log=None):
        if _log is not None:
            self.logger = _log
        else:
            self.logger = logging.getLogger(__name__)

            strHandler = logging.StreamHandler()
            formatter = logging.Formatter(
                    '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
            strHandler.setFormatter(formatter)
            self.logger.addHandler(strHandler)
            self.logger.setLevel(logging.INFO)

        log_dir = self.exp_dir / 'logs'
        os.makedirs(log_dir, exist_ok=True)

        now_str = datetime.datetime.now().__str__().replace(' ','_')
        now_str = now_str.replace(' ','_').replace('-','').replace(':','')

        self.log_file = log_dir / f'LOG_INFO_{now_str}.txt'
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.logger.addHandler(self.log_fileHandler)

        tensorboard_writer_path = log_dir / 'TENSORBOARD'
        self.tensorboard_writer = tensorboardX.SummaryWriter(
            str(tensorboard_writer_path))
        self.logger.info(
            f'==> Tensorboard file location: {tensorboard_writer_path}')

    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info(f'Set network {key}')
            def_file = val['def_file']
            net_opt = val['opt']
            self.optim_params[key] = (
                val['optim_params'] if ('optim_params' in val) else None)
            pretrained_path = val['pretrained'] if ('pretrained' in val) else None
            force = val['force'] if ('force' in val) else False
            self.networks[key] = self.init_network(
                def_file, net_opt, pretrained_path, key, force=force)

    def init_network(self, net_def_file, net_opt, pretrained_path, key, force=False):
        self.logger.info(f'==> Initiliaze network {key} from '
                         f'file {net_def_file} with opts: {net_opt}')
        #if (not os.path.isfile(net_def_file)):
        #    raise ValueError('Non existing file: {0}'.format(net_def_file))
        architecture_name = os.path.basename(net_def_file).split('.py')[0]
        network = architectures.factory(
            architecture_name=architecture_name, opt=net_opt)
        self.logger.info('==> Network architecture:')
        self.logger.info(network)
        if pretrained_path is not None:
            self.load_pretrained(network, pretrained_path, force=force)

        return network

    def load_pretrained(self, network, pretrained_path, force=False):
        all_possible_files = glob.glob(pretrained_path)
        if len(all_possible_files) == 0:
            raise ValueError(f'{pretrained_path}: no such file')
        else:
            pretrained_path = all_possible_files[-1]
        self.logger.info('==> Load pretrained parameters from file %s:' %
                         pretrained_path)

        assert(os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path)
        if pretrained_model['network'].keys() == network.state_dict().keys() or force:
            network.load_state_dict(pretrained_model['network'])
        else:
            self.logger.info(
                f'==> WARNING: network parameters in pre-trained file'
                f' {pretrained_path} do not strictly match')
            #with torch.set_grad_enabled(False):
            for pname, param in network.named_parameters():
                if pname in pretrained_model['network']:
                    self.logger.info(
                        f'==> Copying parameter {pname} from file {pretrained_path}')
                    #breakpoint()
                    param.data.copy_(pretrained_model['network'][pname])

    def init_all_optimizers(self):
        self.optimizers = {}
        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams is not None:
                self.optimizers[key] = self.init_optimizer(
                        self.networks[key], oparams, key)

    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info(
            f'Initialize optimizer: {optim_type} '
            f'with params: {optim_opts} '
            f'for netwotk: {key}')
        if optim_type == 'adam':
            if 'beta' in optim_opts:
                optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                    betas=optim_opts['beta'])
            elif 'weight_decay' in optim_opts:
                optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                    weight_decay=optim_opts['weight_decay'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

        return optimizer

    def init_all_criterions(self):
        self.criterions = {}
        criterions_defs = self.opt.get('criterions')
        if criterions_defs is not None:
            for key, val in criterions_defs.items():
                crit_type = val['ctype']
                crit_opt = val['opt'] if ('opt' in val) else None
                self.logger.info(
                    f'Initialize criterion[{key}]: {crit_type} '
                    f'with options: {crit_opt}')
                self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        if copt is None:
            return getattr(nn, ctype)()
        else:
            return getattr(nn, ctype)(copt)

    def load_to_gpu(self):
        self.device = 'cuda'
        for key, net in self.networks.items():
            self.networks[key] = net.to('cuda')

        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.to('cuda')

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to('cuda')

    def save_checkpoint(self, epoch, suffix='', metric=None):
        for key, net in self.networks.items():
            if self.optimizers[key] is None: continue
            self.save_network(key, epoch, suffix=suffix, metric=metric)
            self.save_optimizer(key, epoch, suffix=suffix)

    def find_most_recent_epoch(self, key, suffix):
        search_patern = self._get_net_checkpoint_filename(key, '*') + suffix
        all_files = glob.glob(search_patern)
        if len(all_files) == 0:
            raise ValueError(f'{search_patern}: no such file.')

        substrings = search_patern.split('*')
        assert(len(substrings) == 2)
        start, end = substrings
        all_epochs = [fname.replace(start,'').replace(end,'') for fname in all_files]
        all_epochs = [int(epoch) for epoch in all_epochs if epoch.isdigit()]
        assert(len(all_epochs) > 0)
        all_epochs = sorted(all_epochs)
        most_recent_epoch = int(all_epochs[-1])
        self.logger.info('Load checkpoint of most recent epoch %s' %
                         str(most_recent_epoch))
        return most_recent_epoch

    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info(f'Load checkpoint of epoch {epoch}')

        for key, net in self.networks.items(): # Load networks
            if self.optim_params[key] is None: continue
            if epoch == '*':
                epoch = self.find_most_recent_epoch(key, suffix)
            self.load_network(key, epoch, suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] is None: continue
                self.load_optimizer(key, epoch, suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] is None: continue

            filename_net = pathlib.Path(
                self._get_net_checkpoint_filename(key, epoch)+suffix)
            if filename_net.is_file(): os.remove(filename_net)

            filename_optim = pathlib.Path(
                self._get_optim_checkpoint_filename(key, epoch)+suffix)
            if filename_optim.is_file(): os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix='', metric=None):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        state = {
            'epoch': epoch,
            'network': self.networks[net_key].state_dict(),
            'metric': metric
        }
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        state = {
            'epoch': epoch,
            'optimizer': self.optimizers[net_key].state_dict()
        }
        torch.save(state, filename)

    def load_network(self, net_key, epoch, suffix=''):
        assert(net_key in self.networks)
        filename = pathlib.Path(
            self._get_net_checkpoint_filename(net_key, epoch)+suffix)
        self.logger.info(f'Loading {filename} for network {net_key}')
        assert filename.is_file()
        checkpoint = torch.load(filename)
        self.networks[net_key].load_state_dict(checkpoint['network'])

    def load_optimizer(self, net_key, epoch,suffix=''):
        assert(net_key in self.optimizers)
        filename = pathlib.Path(
            self._get_optim_checkpoint_filename(net_key, epoch)+suffix)
        assert filename.is_file()
        checkpoint = torch.load(filename)
        self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return str(self.exp_dir / f'{net_key}_net_epoch{epoch}')

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return str(self.exp_dir / f'{net_key}_optim_epoch{epoch}')

    def solve(self, data_loader_train, data_loader_test=None):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info('Training epoch [%3d / %3d]' %
                             (self.curr_epoch + 1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.logger.info('==> Training stats: %s' % train_stats)

            self.save_checkpoint(self.curr_epoch+1) # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch: # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)

            if data_loader_test is not None:
                if isinstance(data_loader_test, (list, tuple)):
                    for i, data_loader_test_this in enumerate(data_loader_test):
                        eval_stats = self.evaluate(data_loader_test_this, idx=i)
                        self.logger.info(
                            '==> Evaluation stats: %s' % eval_stats)
                else:
                    eval_stats = self.evaluate(data_loader_test)
                    self.logger.info('==> Evaluation stats: %s' % eval_stats)
                    self.keep_record_of_best_model(eval_stats, self.curr_epoch)

                self.print_eval_stats_of_best_model()

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info(f'Training: {self.exp_dir.name}')
        self.dloader = data_loader
        self.dataset_train = data_loader.dataset

        for key, network in self.networks.items():
            if self.optimizers[key] is None: network.eval()
            else: network.train()

        dname = ''
        if 'name' in self.dloader.dataset.__dict__.keys():
            dname = self.dloader.dataset.name

        self.logger.info(f'==> Dataset: {dname} [{len(self.dloader)} batches]')
        disp_step = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        train_stats = utils.DAverageMeter('train', self._run)
        self.bnumber = len(data_loader)
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx # batch iteration.
            self.global_iter = self.curr_epoch * len(data_loader) + self.biter
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx+1) % disp_step == 0:
                self.logger.info(
                    '==> Iteration [%3d][%4d / %4d]: %s' %
                    (epoch+1, idx+1, len(data_loader), train_stats))

        train_stats.log()
        self.add_stats_to_tensorboard_writer(
            train_stats.average(), 'train_')

        return train_stats

    def evaluate(self, dloader, idx=None):
        self.logger.info(f'Evaluating: {self.exp_dir.name}')

        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        self.logger.info('==> Dataset: %s [%d batches]' %
                         (dloader.dataset.name, len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        if idx is not None:
            meter_name = f'eval_{idx}'
        else:
            meter_name = 'eval'
        eval_stats = utils.DAverageMeter(meter_name, self._run)
        self.bnumber = len(dloader)
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats)
        eval_stats.log()
        self.add_stats_to_tensorboard_writer(
            eval_stats.average(), 'test_')

        return eval_stats

    def add_stats_to_tensorboard_writer(self, stats, prefix):
        for key, value in stats.items():
            if isinstance(value, dict):
                self.add_stats_to_tensorboard_writer(
                    value, prefix + key + '_')
            else:
                self.tensorboard_writer.add_scalar(
                    prefix + key, value, self.curr_epoch)

    def adjust_learning_rates(self, epoch):
        # filter out the networks that are not trainable and that do
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k:v for k,v in self.optim_params.items()
            if (v is not None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            self.logger.info(f'==> Set to {key} optimizer lr = {lr:.10f}')
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if metric_name not in eval_stats.values:
                raise ValueError(
                    f'The provided metric {metric_name} for keeping the best '
                     'model is not computed by the evaluation routine.')
            metric_val = eval_stats.average()[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(
                    self.curr_epoch+1, suffix='.best', metric=self.max_metric_val)
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info(f'==> Best results w.r.t. {metric_name} '
                             f'metric: epoch: {self.best_epoch+1} - {self.best_stats}')

    def optimizer_zero_grad(self, optimizer):
        if optimizer is not None:
            optimizer.zero_grad()

    def optimizer_step(self, optimizer):
        if optimizer is not None:
            optimizer.step()

    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}
