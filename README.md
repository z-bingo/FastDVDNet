## FastDVDNet
The model file has released at [here](https://github.com/z-bingo/FastDVDNet/tree/master/models)!  
  
TODO LIST: 
1. write the code of single-stage model, single-scale model to verify the superiority of FastDVDNet

## Introduction
This repo. is an unofficial version of [FastDVDNet:ToWards Real-Time Video Denoising Without Explicit Motion Estimation](https://arxiv.org/pdf/1907.01361.pdf) by making use of PyTorch.

## Dataset
The dataset of [Vimeo-90K](http://toflow.csail.mit.edu/) is employed for training, whose size is about 82G. The dataset consists of about 90K videos and all of them include 7 frames with large motion.

You can choose the dataset you like to train or validate the effectiveness of FastDVDNet, such as the dataset indicated in the paper (I don't want to spend more time downloading it so I choose the Vimeo-90K.).

## Requirements
1. PyTorch>=1.0.0
2. Numpy
3. scikti-image
4. tensorboardX (for visualization of loss, PSNR and SSIM)


## Usage
1. Run `pip install -r requirements.txt` firstly to install the packages.
2. `data_provider.py` is the code for loading data from dataset. You can modify the code to adapt your dataset if it is not Vimeo-90K.
3. `train_eval.py` is the code for train and validation process. The validation process is activated by `--eval`, otherwise, it is work on the train mode.
4. Script for restarting the model can be follow,<br>
`CUDA_VISIBLE_DEVICES=1,2 python train_eval.py --cuda -nw 16 --frames 5 -s 96 -bs 64 -lr 1e-4 --restart`
5. As for validation mode,<br>
`CUDA_VISIBLE_DEVICES=1,2 python train_eval.py --cuda --eval`
for this mode, the command `-s` is unavailable, image size is default 448*256.

### Details

```
train_eval.py [-h] [--dataset_path DATASET_PATH] [--txt_path TXT_PATH]
                     [--batch_size BATCH_SIZE] [--frames FRAMES]
                     [--im_size IM_SIZE] [--learning_rate LEARNING_RATE]
                     [--num_worker NUM_WORKER] [--restart] [--eval] [--cuda]
                     [--max_epoch MAX_EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH, -dp DATASET_PATH
                        the path of vimeo-90k
  --txt_path TXT_PATH, -tp TXT_PATH
                        the path of train/eval txt file
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        batch size
  --frames FRAMES, -f FRAMES
  --im_size IM_SIZE, -s IM_SIZE
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --num_worker NUM_WORKER, -nw NUM_WORKER
                        number of workers to load data by dataloader
  --restart, -r         whether to restart the train process
  --eval, -e            whether to work on the eval mode
  --cuda                whether to train the network on the GPU, default is
                        mGPU
  --max_epoch MAX_EPOCH
```

## Results


## References
1. [FastDVDNet:ToWards Real-Time Video Denoising Without Explicit Motion Estimation](https://arxiv.org/pdf/1907.01361.pdf)
2. [DVDNet: A Fast Network For Deep Video Denoising](https://arxiv.org/pdf/1906.11890.pdf)
3. Code of U-Net is based on [my previous work](https://github.com/z-bingo/Recurrent-Fully-Convolutional-Networks/blob/master/U_Net.py)

*Support me by starring or forking this repo., please.*
