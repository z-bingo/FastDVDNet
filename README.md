## FastDVDNet
TODO: Train the network and upload the trained model, and write the documents.

## Introduction
This repo. is an unofficial version of [FastDVDNet:ToWards Real-Time Video Denoising Without Explicit Motion Estimation](https://arxiv.org/pdf/1907.01361.pdf) by making use of PyTorch.

## Dataset
The dataset of [Vimeo-90K](http://toflow.csail.mit.edu/) is employed for training, whose size is about 82G. The dataset consists of about 90K videos and all of them include 7 frames with large motion.

You can choose the dataset you like to train or validate the effectiveness of FastDVDNet, such as the dataset indicated in the paper (I don't want to spend more time downloading it so I choose the Vimeo-90K.).

## Requirements
1. PyTorch>=1.0.0
2. Numpy

## Usage
1. `data_provider.py` is the code for loading data from dataset. You can modify the code to adapt your dataset if it is not Vimeo-90K.
2. `train_eval.py` is the code for train and validation process. The validation process is activated by `--eval`, otherwise, it is work on the train mode.
3. Script for restarting the model can be follow,<br>
`CUDA_VISIBLE_DEVICES=1,2 python train_eval.py --cuda -nw 16 --frames 5 -s 96 -bs 64 -lr 1e-4 --restart`
4. As for validation mode,<br>
`CUDA_VISIBLE_DEVICES=1,2 python train_eval.py --cuda --eval`

## References
1. [FastDVDNet:ToWards Real-Time Video Denoising Without Explicit Motion Estimation](https://arxiv.org/pdf/1907.01361.pdf)
2. [DVDNet: A Fast Network For Deep Video Denoising](https://arxiv.org/pdf/1906.11890.pdf)

*Support me by starring or forking this repo., please.*
