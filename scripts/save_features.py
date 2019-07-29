"""
Extracts and saves features (with a model trained by the lowshot_train_stage1.py
routine) from the images of the ImageNet dataset.

Example of usage:
# Extract features from the validation image split of the Imagenet.
python scripts/save_features.py --config=imagenet_ResNet10CosineClassifier --split='val'
# Extract features from the training image split of the Imagenet.
python scripts/save_features.py --config=imagenet_ResNet10CosineClassifier --split='train'

The config argument specifies the model that will be used.
"""

from __future__ import print_function

import argparse
import os

from low_shot_learning.algorithms.utils.save_features import SaveFeatures
from low_shot_learning.dataloaders.basic_dataloaders import SimpleDataloader
from low_shot_learning.datasets.imagenet_dataset import ImageNet
from low_shot_learning import project_root

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with hyper-parameters of the model that we will use for '
         'extracting features from ImageNet dataset.')
parser.add_argument('--checkpoint', type=int, default=-1,
    help='checkpoint (epoch id) that will be loaded. If a negative value is'
         ' given then the latest existing checkpoint is loaded.')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save2exp', default=False, action='store_true')
parser.add_argument('--feature_name', type=str, default='')
parser.add_argument('--global_pooling', default=False, action='store_true')
args_opt = parser.parse_args()


exp_base_directory = os.path.join(project_root, 'experiments')
exp_directory = os.path.join(exp_base_directory, args_opt.config)

# Load the configuration params of the experiment
exp_config_file = 'config.' + args_opt.config.replace('/', '.')
#print(f'Launching experiment: {exp_config_file}')
config = __import__(exp_config_file, fromlist=['']).config
config['exp_dir'] = exp_directory # where logs, models, etc will be stored.
print(f'Loading experiment {args_opt.config}')
print(f'Generated logs, snapshots, and model files will be stored on {exp_directory}')

if (args_opt.split != 'train') and (args_opt.split != 'val'):
    raise ValueError('Not valid split {0}'.format(args_opt.split))

dataset = ImageNet(
    split=args_opt.split, use_geometric_aug=False, use_color_aug=False)
dloader = SimpleDataloader(
    dataset,
    batch_size=args_opt.batch_size,
    train=False,
    num_workers=args_opt.num_workers)

algorithm = SaveFeatures(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0: # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else '*',
        train=False)

if args_opt.save2exp:
    dst_directory = os.path.join(exp_directory, 'feature_datasets')
else:
    dst_directory = os.path.join(
        project_root, 'datasets', 'feature_datasets', args_opt.config)

if args_opt.feature_name == '':
    args_opt.feature_name = None
else:
    dst_directory = dst_directory + '_' + args_opt.feature_name

algorithm.logger.info(f"==> Destination directory: {dst_directory}")
if (not os.path.isdir(dst_directory)):
    os.makedirs(dst_directory)

dst_filename = os.path.join(
    dst_directory, 'ImageNet_' + args_opt.split + '.h5')

algorithm.logger.info(f"==> dst_filename: {dst_filename}")
algorithm.logger.info(f"==> args_opt.feature_name: {args_opt.feature_name}")
algorithm.logger.info(f"==> args_opt.global_pooling: {args_opt.global_pooling}")

algorithm.save_features(
    dataloader=dloader,
    filename=dst_filename,
    feature_name=args_opt.feature_name,
    global_pooling=args_opt.global_pooling)
