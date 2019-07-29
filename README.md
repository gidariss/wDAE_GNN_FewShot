# *Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning*

The current project page provides [pytorch](http://pytorch.org/) code that implements the following CVPR2019 paper (accepted as oral):   
**Title:**      "Generating Classification Weights with GNN Denoising Autoencoders for Few-Shot Learning"    
**Authors:**     Spyros Gidaris, Nikos Komodakis    
**Code:**        https://github.com/gidariss/wDAE_GNN_FewShot    

**Abstract:**  
Given an initial recognition model already trained on a set of base classes, the goal of this work is to develop a meta-model for few-shot learning. The meta-model, given as input some novel classes with few training examples per class, must properly adapt the existing recognition model into a new model that can correctly classify in a unified way both the novel and the base classes. To accomplish this goal it must learn to output the appropriate classification weight vectors for those two types of classes. To build our meta-model we make use of two main innovations: we propose the use of a Denoising Autoencoder network (DAE) that (during training) takes as input a set of classification weights corrupted with Gaussian noise and learns to reconstruct the target-discriminative classification weights. In this case, the injected noise on the classification weights serves the role of regularizing the weight generating meta-model. Furthermore, in order to capture the co-dependencies between different classes in a given task instance of our meta-model, we propose to implement the DAE model as a Graph Neural Network (GNN). In order to verify the efficacy of our approach, we extensively evaluate it on ImageNet based few-shot benchmarks and we report strong results that surpass prior approaches.


### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Contents:
**(1)** Code for running the ImageNet-based experiments with the wDAE-GNN-based few-shot model.    

**(2)** Code for running the MiniImageNet-based experiments: would be ready soon.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.0.0
* CUDA 10.0 or higher

### Installation

**(1)** Clone the repo:
```bash
$ git clone https://github.com/gidariss/wDAE_GNN_FewShot
```

**(2)** Install this repository and the dependencies using pip:
```bash
$ pip install -e ./wDAE_GNN_FewShot
```

With this, you can edit the wDAE_GNN_FewShot code on the fly and import function
and classes of wDAE_GNN_FewShot in other project as well.

**(3)** Optional. To uninstall this package, run:
```bash
$ pip uninstall wDAE_GNN_FewShot
```

**(4)** Create *dataset* and *experiment* directories:
```bash
$ cd wDAE_GNN_FewShot
$ mkdir ./datasets
$ mkdir ./experiments
```

You can take a look at the [Dockerfile](./Dockerfile) if you are uncertain about steps to install this project.

## Running experiments on the ImageNet based few-shot benchmark

Here I provide instructions for training and evaluating our method on the ImageNet based low-shot benchmark proposed by Bharath and Girshick [1].

**(1)** Download the ImageNet dataset and set in [imagenet_dataset.py](https://github.com/gidariss/wDAE_GNN_FewShot/blob/master/low_shot_learning/datasets/imagenet_dataset.py#L19) the path to where the dataset resides in your machine.

**(2)** Train a ResNet10 based recognition model with cosine similarity-based classifier [3]:     
```bash
$ cd wDAE_GNN_FewShot # enter the wDAE_GNN_FewShot directory.
$ python scripts/lowshot_train_stage1.py --config=imagenet_ResNet10CosineClassifier
```    
You can download the already trained by us recognition model from [here](https://github.com/gidariss/wDAE_GNN_FewShot/releases/download/0.1/imagenet_ResNet10CosineClassifier.zip). In that case, place the model inside the './experiments' directory with the name './experiments/imagenet_ResNet10CosineClassifier'.     
```bash
# Run from the wDAE_GNN_FewShot directory
$ cd ./experiments
$ wget https://github.com/gidariss/wDAE_GNN_FewShot/releases/download/0.1/imagenet_ResNet10CosineClassifier.zip
$ unzip imagenet_ResNet10CosineClassifier.zip
$ cd ..
```

**(3)** Extract and save the ResNet10 features (with the above model; see step (2)) from images of the ImageNet dataset:    
```bash
# Run from the wDAE_GNN_FewShot directory
# Extract features from the validation image split of the Imagenet.
$ python scripts/save_features.py --config=imagenet_ResNet10CosineClassifier --split='val'
# Extract features from the training image split of the Imagenet.
$ python scripts/save_features.py --config=imagenet_ResNet10CosineClassifier --split='train'
```   
The features will be saved on './datasets/feature_datasets/imagenet_ResNet10CosineClassifier'.
You can download the pre-computed features from [here](https://mega.nz/#!bsVlzQBR!MNADfBM4JX2KgWG13oL0pXhHCQqvkPRD4MfP_aUOtXg). In that case, place the downloaded features in './datasets/' with the following structure:
```
# Features of the validation images of ImageNet.    
./datasets/feature_datasets/imagenet_ResNet10CosineClassifier/ImageNet_val.h5
# Features of the training images of ImageNet.
./datasets/feature_datasets/imagenet_ResNet10CosineClassifier/ImageNet_train.h5
```


**(4)** Train the Graph Neural Network Denoising AutoEncoder few-shot model (wDAE_GNN):
```bash
# Run from the wDAE_GNN_FewShot directory
# Training the wDAE-GNN few-shot model.
$ python scripts/lowshot_train_stage2.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN
```    
The model will be saved on 'wDAE_GNN_FewShot/experiments/imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN'.
Otherwise, you can download the pre-trained few-shot model from
[here](https://github.com/gidariss/wDAE_GNN_FewShot/releases/download/0.1/imagenet_ResNet10CosineClassifier_wDAE_GNN.zip).
In that case, place the downloaded model in
'wDAE_GNN_FewShot/experiments/imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN'.
```bash
# Run from the wDAE_GNN_FewShot directory
$ cd experiments # enter the wDAE_GNN_FewShot directory.
$ mkdir imagenet_wDAE
$ cd imagenet_wDAE
$ wget https://github.com/gidariss/wDAE_GNN_FewShot/releases/download/0.1/imagenet_ResNet10CosineClassifier_wDAE_GNN.zip
$ unzip imagenet_ResNet10CosineClassifier_wDAE_GNN.zip
$ cd ../../
```


**(5)** Evaluate the above trained model:   
```bash
# Run from the wDAE_GNN_FewShot directory
# Evaluate the model on the 1-shot setting.
$ python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=1 --step_size=1.0
# Expected output:
# ==> Top 5 Accuracies:      [Novel: 47.99 | Base: 93.39 | All 59.02 ]

# Evaluate the model on the 2-shot setting.
$ python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=2 --step_size=1.0
# Expected output:
# ==> Top 5 Accuracies:      [Novel: 59.54 | Base: 93.39 | All 66.22 ]

# Evaluate the model on the 5-shot setting.
$ python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=5 --step_size=0.6
# Expected output:
# ==> Top 5 Accuracies:      [Novel: 70.23 | Base: 93.44 | All 73.20 ]

# Evaluate the model on the 10-shot setting.
$ python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=10 --step_size=0.4
# Expected output:
# ==> Top 5 Accuracies:      [Novel: 74.95 | Base: 93.37 | All 76.09 ]

# Evaluate the model on the 20-shot setting.
$ python scripts/lowshot_evaluate.py --config=imagenet_wDAE/imagenet_ResNet10CosineClassifier_wDAE_GNN --testset --nexemplars=20 --step_size=0.2
# Expected output:
# ==> Top 5 Accuracies:      [Novel: 77.77 | Base: 93.33 | All 77.54 ]
```

## Experimental results on the ImageNet based Low-shot benchmark

Here I provide the experiment results of the few-shot model trained with this code on the ImageNet-based low-shot [1] using the evaluation metrics proposed by [2].
Note that after cleaning and refactoring the implementation code of the paper
and re-running the experiments, the results that we got are slightly different.

### Top-5 classification accuracy of wDAE-GNN model.
| wDAE-GNN                             | Novel           | All             |
| ------------------------------------ | ---------------:|----------------:|
|  1-shot results                      | 47.99%          | 59.02%          |
|  2-shot results                      | 59.54%          | 66.22%          |
|  5-shot results                      | 70.23%          | 73.20%          |
| 10-shot results                      | 74.95%          | 76.09%          |
| 20-shot results                      | 77.77%          | 77.54%          |

### References
```
[1] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features.
[2] Y.-X. Wang and R. Girshick, M. Hebert, B. Hariharan. Low-shot learning from imaginary data.
[3] S. Gidaris and N. Komodakis. Dynamic few-shot visual learning without forgetting.
```
