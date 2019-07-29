"""Train the wDAE-GNN few-shot model on the the low-shot Imagenet dataset[*].

Example of usage:
python scripts/lowshot_train_stage2.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN
imagenet_ResNet10CosineClassifier_wDAE_GNN

All the configuration files above (i.e., specified by the --config argument) are
placed on the directory ./config .

[*] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
"""

from __future__ import print_function

import argparse
import os

from low_shot_learning.algorithms.fewshot.imagenet_lowshot import ImageNetLowShot
from low_shot_learning.dataloaders.dataloader_fewshot import FewShotDataloader, LowShotDataloader
from low_shot_learning.datasets.imagenet_dataset import ImageNetLowShotFeatures
from low_shot_learning import project_root

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment')
parser.add_argument('--checkpoint', type=int, default=0,
    help='checkpoint (epoch id) that will be loaded. If a negative value is '
         'given then the latest existing checkpoint is loaded.')
parser.add_argument('--num_workers', type=int, default=0,
    help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=200,
    help='display step during training')
args_opt = parser.parse_args()

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

config['disp_step'] = args_opt.disp_step
algorithm = ImageNetLowShot(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0: # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if  (args_opt.checkpoint > 0) else '*',
        train=True)

# Set the train dataset and the corresponding data loader.
data_train_opt = config['data_train_opt']
feat_dataset_train = ImageNetLowShotFeatures(
    data_dir=data_train_opt['data_dir'],
    image_split='train',
    phase='train')
dloader_train = FewShotDataloader(
    dataset=feat_dataset_train,
    nKnovel=data_train_opt['nKnovel'],
    nKbase=data_train_opt['nKbase'],
    nExemplars=data_train_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_train_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_train_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_train_opt['batch_size'],
    num_workers=args_opt.num_workers,
    epoch_size=data_train_opt['epoch_size'], # num of batches per epoch
)

feat_data_train = ImageNetLowShotFeatures(
    data_dir=data_train_opt['data_dir'], image_split='train', phase='val')
feat_data_test = ImageNetLowShotFeatures(
    data_dir=data_train_opt['data_dir'], image_split='val', phase='val')
dloader_test = LowShotDataloader(
    feat_data_train, feat_data_test,
    nExemplars=data_train_opt['nExemplars'],
    batch_size=200,
    num_workers=0)

algorithm.solve(dloader_train, dloader_test)
