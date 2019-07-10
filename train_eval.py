import torch
import torch.nn as nn
import numpy as np
from arch.FastDVDNet import FastDVDNet
from utils.data_utils import *
from utils.file_utils import *
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from data_provider import Video_Provider
import os, sys, shutil
import torch.optim as optim
import time

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-dp', default='/media/sde/zb/rnn-cnn/vimeo_septuplet/sequences', help='the path of vimeo-90k')
    parser.add_argument('--txt_path', '-tp', default='/media/sde/zb/rnn-cnn/vimeo_septuplet', help='the path of train/eval txt file')
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='batch size')
    parser.add_argument('--frames', '-f', default=5, type=int)
    parser.add_argument('--im_size', '-s', default=96, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--num_worker', '-nw', default=4, type=int, help='number of workers to load data by dataloader')
    parser.add_argument('--restart', '-r', action='store_true', help='whether to restart the train process')
    parser.add_argument('--eval', '-e', action='store_true', help='whether to work on the eval mode')
    parser.add_argument('--cuda', action='store_true', help='whether to train the network on the GPU, default is mGPU')
    parser.add_argument('--max_epoch', default=100, type=int)
    return parser.parse_args()

def train(args):
    data_set = Video_Provider(
        base_path=args.dataset_path,
        txt_file=os.path.join(args.txt_path, 'sep_trainlist.txt'),
        im_size=args.im_size,
        frames=args.frames
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker
    )
    #
    model = FastDVDNet(
        in_frames=args.frames
    )
    # run on the GPU
    if args.cuda:
        model = nn.DataParallel(model.cuda())

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # whether to load the existent model
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if args.restart:
        rm_sub_files('./models')
        epoch = 0
        global_iter = 0
        best_loss = np.inf
        print('Start the train process.')
    else:
        try:
            state = load_checkpoint('./models', is_best=True)
            epoch = state['epoch']
            global_iter = state['global_iter']
            best_loss = state['best_loss']
            optimizer.load_state_dict(state['optimizer'])
            model.load_state_dict(state['state_dict'])
            print('Model load OK at global_iter {}, epoch {}.'.format(global_iter, epoch))
        except:
            epoch = 0
            global_iter = 0
            best_loss = np.inf
            print('There is no any model to load, restart the train process.')

    #
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    log_writer = SummaryWriter('./logs')

    loss_func = nn.MSELoss()
    t = time.time()
    loss_temp = 0
    psnr_temp = 0
    ssim_temp = 0
    model.train()
    for e in range(epoch, args.max_epoch):
        for iter, (data, gt) in enumerate(data_loader):
            if args.cuda:
                data = data.cuda()
                gt = gt.cuda()

            pred = model(data)
            loss = loss_func(gt, pred)
            global_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(pred, gt)
            ssim = calculate_ssim(pred, gt)

            print(
                '{:6d} epoch: {:2d}, loss: {:.4f}, PSNR: {:.2f}dB, SSIM: {:.4f}, time: {:.2}S.'.format(
                    global_iter, epoch, loss, psnr, ssim, time.time() - t
                )
            )
            log_writer.add_scalar('loss', loss, global_iter)
            log_writer.add_scalar('psnr', psnr, global_iter)
            log_writer.add_scalar('ssim', ssim, global_iter)
            t = time.time()

            psnr_temp += psnr
            ssim_temp += ssim

            loss_temp += loss
            if global_iter % 100 == 0:
                loss_temp /= 100
                psnr_temp /= 100
                ssim_temp /= 100
                is_best = True if loss_temp < best_loss else False
                best_loss = min(best_loss, loss_temp)
                state = {
                    'state_dict': model.state_dict(),
                    'epoch': e,
                    'global_iter': global_iter,
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }
                save_checkpoint(state, global_iter, path='./models', is_best=is_best, max_keep=20)

                t = time.time()
                loss_temp, psnr_temp, ssim_temp = 0, 0, 0


def eval(args):
    from torchvision.transforms import transforms
    from PIL import Image
    data_set = Video_Provider(
        base_path=args.dataset_path,
        txt_file=os.path.join(args.txt_path, 'sep_testlist.txt'),
        im_size=256,
        frames=args.frames
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_worker
    )
    #
    model = FastDVDNet(
        in_frames=args.frames
    )
    # run on the GPU
    if args.cuda:
        model = nn.DataParallel(model.cuda())

    state = load_checkpoint('./models', is_best=True)
    model.load_state_dict(state['state_dict'])
    model.eval()

    rm_sub_files('./eval_images')
    trans = transforms.ToPILImage()
    for i, (data, gt) in enumerate(data_loader):
        if i > 50:
            break
        if args.cuda:
            data = data.cuda()
            gt = gt.cuda()

        pred = model(data)
        psnr_pred = calculate_psnr(pred, gt)
        ssim_pred = calculate_ssim(pred, gt)

        if args.cuda:
            data = data.cpu().numpy()
            gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()

        trans((data[0, 6:9, ...] * 255.0).as_type(np.uint8)).save('{}_noisy.png'.format(i), quality=100)
        trans((gt * 255.0).as_type(np.uint8)).save('{}_gt.png'.format(i), quality=100)
        trans((pred * 255.0).as_type(np.uint8)).save('{}_pred_{:.2f}dB_{:.4f}.png'.format(i, psnr_pred, ssim_pred), quality=100)



if __name__ == '__main__':
    args = args_parser()
    print(args)
    if not args.eval:
        train(args)
    else:
        eval(args)