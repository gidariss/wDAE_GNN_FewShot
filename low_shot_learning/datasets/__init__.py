from low_shot_learning.datasets.imagenet_dataset import ImageNet
from low_shot_learning.datasets.imagenet_dataset import ImageNetLowShot
from low_shot_learning.datasets.imagenet_dataset import ImageNetFeatures
from low_shot_learning.datasets.imagenet_dataset import ImageNetLowShotFeatures
from low_shot_learning.datasets.mini_imagenet_dataset import MiniImageNet
from low_shot_learning.datasets.mini_imagenet_dataset import MiniImageNet80x80
from low_shot_learning.datasets.mini_imagenet_dataset import MiniImageNetFeatures


def dataset_factory(dataset_name, *args, **kwargs):
    datasets_collection = {}
    datasets_collection['MiniImageNet'] = MiniImageNet
    datasets_collection['MiniImageNet80x80'] = MiniImageNet80x80
    datasets_collection['MiniImageNetFeatures'] = MiniImageNetFeatures
    datasets_collection['ImageNet'] = ImageNet
    datasets_collection['ImageNetLowShot'] = ImageNetLowShot
    datasets_collection['ImageNetFeatures'] = ImageNetFeatures
    datasets_collection['ImageNetLowShotFeatures'] = ImageNetLowShotFeatures

    return datasets_collection[dataset_name](*args, **kwargs)
