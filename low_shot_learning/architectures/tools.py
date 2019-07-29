import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        # initialize to the identity transform
        weight = torch.FloatTensor(num_features).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


def cosine_fully_connected_layer(x_in, weight, scale=None, bias=None):
    assert(x_in.dim() == 2)
    assert(weight.dim() == 2)
    assert(x_in.size(1) == weight.size(0))

    x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)
    weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out


def batch_cosine_fully_connected_layer(x_in, weight, scale=None, bias=None):
    """
    Args:
        x_in: a 3D tensor with shape
            [meta_batch_size x num_examples x num_features_in]
        weight: a 3D tensor with shape
            [meta_batch_size x num_features_in x num_features_out]
        scale: (optional) a scalar value
        bias: (optional) a 1D tensor with shape [num_features_out]

    Returns:
        x_out: a 3D tensor with shape
            [meta_batch_size x num_examples x num_features_out]
    """

    assert(x_in.dim() == 3)
    assert(weight.dim() == 3)
    assert(x_in.size(0) == weight.size(0))
    assert(x_in.size(2) == weight.size(1))

    x_in = F.normalize(x_in, p=2, dim=2, eps=1e-12)
    weight = F.normalize(weight, p=2, dim=1, eps=1e-12)

    x_out = torch.bmm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale

    if bias is not None:
        x_out = x_out + bias

    return x_out


class CosineFullyConnectedLayer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        scale=20.0,
        per_plane=False,
        learn_scale=True,
        bias=False):
        super(CosineFullyConnectedLayer, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learn_scale = learn_scale
        self.per_plane = per_plane

        weight = torch.FloatTensor(num_inputs, num_outputs).normal_(
            0.0, np.sqrt(2.0/num_inputs))
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_outputs).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        if scale:
            num_scale_values = num_outputs if per_plane else 1
            scale = torch.FloatTensor(num_scale_values).fill_(scale)
            self.scale = nn.Parameter(scale, requires_grad=learn_scale)
        else:
            self.scale = None

    def forward(self, x_in):
        assert(x_in.dim() == 2)
        return cosine_fully_connected_layer(
            x_in, self.weight, scale=self.scale, bias=self.bias)

    def extra_repr(self):
        s = 'num_inputs={0}, num_classes={1}'.format(
            self.num_inputs, self.num_outputs)

        if self.scale is not None:
            if self.per_plane:
                s += 'num_scales={0} (learnable={1})'.format(
                    self.num_outputs, self.learn_scale)
            else:
                s += 'num_scales={0} (value={1} learnable={2})'.format(
                    1, self.scale.item(), self.learn_scale)

        if self.bias is None:
            s += ', bias=False'

        return s


def global_pooling(x, pool_type):
    assert(x.dim() == 4)
    if pool_type == 'max':
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif pool_type == 'avg':
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError('Unknown pooling type.')


class GlobalPooling(nn.Module):
    def __init__(self, pool_type):
        super(GlobalPooling, self).__init__()
        assert(pool_type == 'avg' or pool_type == 'max')
        self.pool_type = pool_type

    def forward(self, x):
        return global_pooling(x, pool_type=self.pool_type)
