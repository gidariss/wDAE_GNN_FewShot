config = {}
# set the parameters related to the training and testing set

nKbase = 389

data_train_opt = {}
data_train_opt['nKnovel'] = 0
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = 0
data_train_opt['nTestNovel'] = 0
data_train_opt['nTestBase'] = 400
data_train_opt['batch_size'] = 1
data_train_opt['epoch_size'] = 4000
config['data_train_opt'] = data_train_opt

config['max_num_epochs'] = 100

networks = {}
net_optim_paramsF = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True,
    'LUT_lr':[(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]}
networks['feature_extractor'] = {
    'def_file': 'feature_extractors.resnet_feat.py', 'pretrained': None,
    'opt': {'userelu': False, 'restype': 'ResNet10'},
    'optim_params': net_optim_paramsF}

net_optim_paramsC = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True,
    'LUT_lr':[(30, 0.1), (60, 0.01), (90, 0.001), (100, 0.0001)]}
net_optionsC = {
    'num_features':512,
    'num_classes': 1000,
    'global_pooling': False,
    'scale_cls': 10,
    'learn_scale': True}
networks['classifier'] = {
    'def_file': 'classifiers.cosine_classifier_with_weight_generator.py',
    'pretrained': None, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions
