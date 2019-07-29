config = {}
# set the parameters related to the training and testing set

nKbase = 389
nKnovel = 200
nExemplars = 1

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel
data_train_opt['nTestBase'] = nKnovel
data_train_opt['batch_size'] = 4
data_train_opt['epoch_size'] = 4000
data_train_opt['data_dir'] = './datasets/feature_datasets/imagenet_ResNet10CosineClassifier'

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 15

num_features = 512

networks = {}
networks['feature_extractor'] = {
	'def_file': 'feature_extractors.dumb_feat', 'pretrained': None,
	'opt': {'dropout': 0},  'optim_params': None }

net_optim_paramsC = {
	'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
	'nesterov': True, 'LUT_lr':[(10, 0.01), (15, 0.001)]}
pretrainedC = './experiments/imagenet_ResNet10CosineClassifier/classifier_net_epoch100'

net_optionsC = {
	'num_features': num_features,
	'num_classes': 1000,
	'global_pooling': False,
	'scale_cls': 10.0,
	'learn_scale': True,
	'dae_config': {
		'gaussian_noise': 0.08,
		'comp_reconstruction_loss': True,
		'targets_as_input': False,
		'dae_type': 'RelationNetBasedGNN',
		'num_layers': 2,
		'num_features_input': num_features,
		'num_features_output': 2 * num_features,
		'num_features_hidden': 3 * num_features,
		'update_dropout': 0.7,

		'nun_features_msg': 3 * num_features,
		'aggregation_dropout': 0.7,
		'topK_neighbors': 10,
		'temperature': 5.0,
		'learn_temperature': False,
	},
}
networks['classifier'] = {
	'def_file': 'classifiers.cosine_classifier_with_DAE_weight_generator',
	'pretrained': pretrainedC, 'opt': net_optionsC,
	'optim_params': net_optim_paramsC}
config['networks'] = networks

config['criterions'] = {}

config['reconstruction_loss_coef'] = 1.0
config['classification_loss_coef'] = 1.0
