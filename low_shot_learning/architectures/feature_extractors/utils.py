import torch.nn as nn


class SequentialFeatureExtractorAbstractClass(nn.Module):
    def __init__(self, all_feat_names, feature_blocks):
        super(SequentialFeatureExtractorAbstractClass, self).__init__()

        assert(isinstance(feature_blocks, list))
        assert(isinstance(all_feat_names, list))
        assert(len(all_feat_names) == len(feature_blocks))

        self.all_feat_names = all_feat_names
        self._feature_blocks = nn.ModuleList(feature_blocks)


    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = (
            [self.all_feat_names[-1],] if out_feat_keys is None else
            out_feat_keys)

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')

        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. '
                    'Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError(
                    'Duplicate output feature key: {0}.'.format(key))

    	# Find the highest output feature in `out_feat_keys
        max_out_feat = max(
            [self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward the image `x` through the network and output the asked features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. If out_feat_keys is None (
                DEFAULT) then the last feature of the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat+1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = (out_feats[0] if len(out_feats) == 1 else out_feats)

        return out_feats
