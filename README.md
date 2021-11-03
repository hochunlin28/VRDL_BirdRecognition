# VRDL_BirdRecognition
This is homework 1 of VRDL class. I use efficientNet to train the model

* main.py: the trainig code

* inference.py: the testing code of particular presaved model
 
* savedModel.pt: the final model I trained 


## 1. Specification (Environment Settings)
```
pip install -r requirements.txt
```
> use this command to construct the environment

Q1: If it cannot download efficientNet, it have to download from github:

> git clone https://github.com/lukemelas/EfficientNet-PyTorch
> 
> cd EfficientNet-Pytorch
> 
> pip install -e .

reference website: https://www.cnpython.com/pypi/efficientnet-pytorch

Q2: If it cannot use cuda, you must download it in pytorch website via below command:

> conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pytorch download website: https://pytorch.org/get-started/locally/

## 2. Training code
use below command to start training
> python main.py

### data augmentation 
use transforom.compose 

## 3. Evaluation code

## 4. Pre-trained models
