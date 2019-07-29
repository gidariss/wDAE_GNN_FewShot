"""Evaluates a fewshot recognition models on the low-shot Imagenet dataset[*]
using the improved evaluation metrics proposed by Wang et al [**].

Example of usage:
# Evaluate the model on the 1-shot setting.
python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=1 --step_size=1.0
==> Top 5 Accuracies:      [Novel: 47.98 | Base: 93.40 | All 58.99 | Novel vs All 41.16 | Base vs All 87.28 | All prior 57.84]

# Evaluate the model on the 2-shot setting.
python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=2 --step_size=1.0
==> Top 5 Accuracies:      [Novel: 59.52 | Base: 93.41 | All 66.20 | Novel vs All 53.40 | Base vs All 86.51 | All prior 64.87]

# Evaluate the model on the 5-shot setting.
python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=5 --step_size=0.6
==> Top 5 Accuracies:      [Novel: 70.21 | Base: 93.41 | All 73.20 | Novel vs All 65.84 | Base vs All 84.87 | All prior 71.87]

# Evaluate the model on the 10-shot setting.
python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=10 --step_size=0.4
==> Top 5 Accuracies:      [Novel: 74.94 | Base: 93.36 | All 76.08 | Novel vs All 71.74 | Base vs All 82.97 | All prior 75.13]

# Evaluate the model on the 20-shot setting.
python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=20 --step_size=0.2
==> Top 5 Accuracies:      [Novel: 77.77 | Base: 93.33 | All 77.53 | Novel vs All 75.36 | Base vs All 80.98 | All prior 77.11]

The config argument specifies the model that will be evaluated.

[*] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[**] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
"""

from __future__ import print_function

import argparse
import os

from low_shot_learning.algorithms.fewshot.imagenet_lowshot import ImageNetLowShot
from low_shot_learning.dataloaders.dataloader_fewshot import LowShotDataloader
from low_shot_learning.datasets.imagenet_dataset import ImageNetLowShotFeatures
from low_shot_learning import project_root


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment')
parser.add_argument('--checkpoint', type=int, default=-1,
    help='checkpoint (epoch id) that will be loaded. If a negative value is '
         'given then the latest existing checkpoint is loaded.')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--testset', default=False, action='store_true',
    help='If True, the model is evaluated on the test set of ImageNetLowShot. '
         'Otherwise, the validation set is used for evaluation.')
parser.add_argument('--nepisodes', type=int, default=100,
    help='the number of evaluation experiments that will run before computing '
         'the average performance.')
parser.add_argument('--prior', type=float, default=0.7)
parser.add_argument('--nexemplars', type=int, default=-1)
parser.add_argument('--last', default=False, action='store_true')
parser.add_argument('--workspace', default=False, action='store_true')
parser.add_argument('--step_size', default=1.0, type=float)
args_opt = parser.parse_args()
#args_opt.testset = True

exp_config_file = os.path.join(project_root, 'config', args_opt.config + '.py')
exp_base_directory = os.path.join(project_root, 'experiments')
exp_directory = os.path.join(exp_base_directory, args_opt.config)

# Load the configuration params of the experiment
exp_config_file = 'config.' + args_opt.config.replace('/', '.')
#print(f'Launching experiment: {exp_config_file}')
config = __import__(exp_config_file, fromlist=['']).config
config['exp_dir'] = exp_directory # where logs, models, etc will be stored.
print(f'Loading experiment {args_opt.config}')
print(f'Generated logs, snapshots, and model files will be stored on {exp_directory}')

if args_opt.step_size != 1.0:
    config['networks']['classifier']['opt']['dae_config']['step_size'] = args_opt.step_size

algorithm = ImageNetLowShot(config)
if args_opt.cuda: # enable cuda.
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0: # load checkpoint.
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else '*',
        train=False,
        suffix=('' if args_opt.last else '.best'))

# Prepare the datasets and the the dataloader.
nExemplars = data_train_opt = config['data_train_opt']['nExemplars']
if args_opt.nexemplars > 0:
    nExemplars = args_opt.nexemplars

eval_phase = 'test' if args_opt.testset else 'val'
data_train_opt = config['data_train_opt']
feat_data_train = ImageNetLowShotFeatures(
    data_dir=data_train_opt['data_dir'], image_split='train', phase=eval_phase)
feat_data_test = ImageNetLowShotFeatures(
    data_dir=data_train_opt['data_dir'], image_split='val', phase=eval_phase)
data_loader = LowShotDataloader(
    feat_data_train, feat_data_test,
    nExemplars=nExemplars, batch_size=1000, num_workers=1)

results = algorithm.evaluate(
    data_loader,
    num_eval_exp=args_opt.nepisodes,
    prior=args_opt.prior,
    suffix='best')

algorithm.logger.info('==> algorithm_type: {0}'.format('ImageNetLowShot'))
algorithm.logger.info('==> nExemplars: {0}'.format(nExemplars))
algorithm.logger.info('==> num episodes: {0}'.format(args_opt.nepisodes))
algorithm.logger.info('==> eval_phase: {0}'.format(eval_phase))
algorithm.logger.info('==> step_size: {0}'.format(args_opt.step_size))
algorithm.logger.info('==> results: {0}'.format(results))
