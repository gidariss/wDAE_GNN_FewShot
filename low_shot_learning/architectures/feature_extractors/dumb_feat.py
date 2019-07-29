import torch
import torch.nn as nn


class DumbFeat(nn.Module):
    def __init__(self, dropout):
        super(DumbFeat, self).__init__()

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout, inplace=False)
        else:
            self.dropout = None

    def forward(self, x):

        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        assert(x.dim()==2)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


def create_model(opt):
    dropout = opt['dropout'] if ('dropout' in opt) else 0.0
    return DumbFeat(dropout=dropout)
