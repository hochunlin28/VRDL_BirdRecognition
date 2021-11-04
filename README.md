# VRDL_BirdRecognition

This is homework 1 of VRDL class. I use efficientNet to train the model.

* main.py: the trainig code

* inference.py: the testing code of particular presaved model
 
* savedModel.pt: the final model I trained 

* training_labels.txt: include the training image name and its corresponding label

* testing_img_order.txt: the testing image name's order for generating the corresponding submission


## Specification (Environment Settings)
To install requirements:
```
pip install -r requirements.txt
```


Q1: If it cannot download efficientNet, it have to download from github:

```
git clone https://github.com/lukemelas/EfficientNet-PyTorch

cd EfficientNet-Pytorch

pip install -e .
```

> reference website: https://www.cnpython.com/pypi/efficientnet-pytorch

Q2: If it cannot use cuda, you must download it in pytorch website via below command:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

> pytorch download website: https://pytorch.org/get-started/locally/

## Training code
To train the model, run this command:
```
python main.py
```

> hyperparameter settings:
>    
>    - model : efficientnet
>    
>    - learning rate: 0.0002
>    
>    - optimizer: Adam
>      
>    - loss function: cross entropy loss
>    
>    - batch size: 22
>    
>    - epoch: 50 

## Evaluation code
To evaluate model by given testing data, run:
```
python inference.py
```
> After running this code, the program will load the "saveModel.pt" model to evaluate the answer. As the program finished, it wiil produce the "answer.txt" file.

## Pre-trained models / weights
We used effiecientnet pretrained weight (efficientnet-b1) to train
- Can download weight from here : https://github.com/lukemelas/EfficientNet-PyTorch/releases?fbclid=IwAR3kk1dW3WEduBUa5b_cRs-rwfI826_7tkf7BUTV2xOVX1fdL5TFdn_bngE
```
In main.py:

model = EfficientNet.from_pretrained('efficientnet-b1')
```
My saved model downloadable link: https://drive.google.com/file/d/1H1CGvn_vzVStxD4Nzx5wyp-7YQtM8Sb8/view?usp=sharing

## Results
My model achieves the following performance on :
### [2021 VRDL HW1 of CodaLab Competion](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)
| Model name         | Accuracy  | StudentID | Name  |
| ------------------ |-----------|-----------|-------|
| My model           |  70.821%  | 310551098 | 林和俊 | 
