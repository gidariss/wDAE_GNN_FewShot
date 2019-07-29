import math
import torch
import torch.nn as nn

import low_shot_learning.architectures.feature_extractors.utils as utils
import low_shot_learning.architectures.tools as tools


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()

        self.equalInOut = (in_planes == out_planes and stride == 1)

        self.convResidual = nn.Sequential()

        if self.equalInOut:
            self.convResidual.add_module('bn1', nn.BatchNorm2d(in_planes))
            self.convResidual.add_module('relu1', nn.ReLU(inplace=True))
        self.convResidual.add_module(
            'conv1',
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False))

        self.convResidual.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.convResidual.add_module('relu2', nn.ReLU(inplace=True))
        self.convResidual.add_module(
            'conv2',
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False))

        if dropRate > 0:
            self.convResidual.add_module('dropout', nn.Dropout(p=dropRate))

        if self.equalInOut:
            self.convShortcut = nn.Sequential()
        else:
            self.convShortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        return self.convShortcut(x) + self.convResidual(x)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate):

        layers = []
        for i in range(nb_layers):
            in_planes_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(
                block(in_planes_arg, out_planes, stride_arg, dropRate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResnet(utils.SequentialFeatureExtractorAbstractClass):
    def __init__(
        self,
        depth,
        widen_factor=1,
        dropRate=0.0,
        pool='avg',
        extra_block=False,
        block_strides=[2, 2, 2, 2]):
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock

        all_feat_names = []
        feature_blocks = []

        # 1st conv before any network block
        conv1 = nn.Sequential()
        conv1.add_module(
            'Conv',
            nn.Conv2d(3, nChannels[0], kernel_size=3, padding=1, bias=False))
        conv1.add_module('BN', nn.BatchNorm2d(nChannels[0]))
        conv1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(conv1)
        all_feat_names.append('conv1')

        # 1st block.
        block1 = nn.Sequential()
        block1.add_module(
            'Block',
            NetworkBlock(
                n, nChannels[0], nChannels[1], block,
                block_strides[0], dropRate))
        block1.add_module('BN', nn.BatchNorm2d(nChannels[1]))
        block1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block1)
        all_feat_names.append('block1')

        # 2nd block.
        block2 = nn.Sequential()
        block2.add_module(
            'Block',
            NetworkBlock(
                n, nChannels[1], nChannels[2], block,
                block_strides[1], dropRate))
        block2.add_module('BN', nn.BatchNorm2d(nChannels[2]))
        block2.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block2)
        all_feat_names.append('block2')

        # 3rd block.
        block3 = nn.Sequential()
        block3.add_module(
            'Block',
            NetworkBlock(
                n, nChannels[2], nChannels[3], block,
                block_strides[2], dropRate))
        block3.add_module('BN', nn.BatchNorm2d(nChannels[3]))
        block3.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block3)
        all_feat_names.append('block3')

        # extra block.
        if extra_block:
            block4 = nn.Sequential()
            block4.add_module(
                'Block',
                NetworkBlock(
                    n, nChannels[3], nChannels[3], block,
                    block_strides[3], dropRate))
            block4.add_module('BN', nn.BatchNorm2d(nChannels[3]))
            block4.add_module('ReLU', nn.ReLU(inplace=True))
            feature_blocks.append(block4)
            all_feat_names.append('block4')

        # global average pooling and classifier_type
        assert(pool == 'none' or pool == 'avg' or pool == 'max')
        if pool == 'max' or pool == 'avg':
            feature_blocks.append(tools.GlobalPooling(pool_type=pool))
            all_feat_names.append('GlobalPooling')

        super(WideResnet, self).__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def create_model(opt):
    depth = opt['depth']
    widen_factor = opt['widen_Factor']
    dropRate = opt['dropRate'] if ('dropRate' in opt) else 0.0
    pool = opt['pool'] if ('pool' in opt) else 'avg'
    extra_block = opt['extra_block'] if ('extra_block' in opt) else False
    block_strides = opt['strides'] if ('strides' in opt) else None

    if block_strides is None:
        block_strides = [2] * 4

    return WideResnet(
        depth, widen_factor, dropRate, pool, extra_block, block_strides)


if __name__ == '__main__':
    opt = {}
    opt['depth'] = 28
    opt['widen_Factor'] = 10
    opt['dropRate'] = 0.0
    opt['extra_block'] = False
    opt['pool'] = 'none'
    model = create_model(opt)
    print(model)

    batch_size = 1
    image_size = 80
    img = torch.FloatTensor(batch_size, 3, image_size, image_size).normal_()
    features = model(img, model.all_feat_names)
    for feature, feature_name in zip(features, model.all_feat_names):
        print('Feature {0}: size {1}, mean {2}, std {3}'.format(
            feature_name, feature.size(), feature.mean().item(),
            feature.std().item()))

    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            count += parameter.numel()

    print(count)
