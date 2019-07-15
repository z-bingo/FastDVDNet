import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import os

import numpy as np
from PIL import Image

class Video_Provider(Dataset):
    def __init__(self, base_path, txt_file, im_size=96, frames=5):
        super(Video_Provider, self).__init__()
        self.base_path = base_path
        self.txt_file = open(txt_file, 'r').readlines()
        self.im_size = im_size
        self.trans = transforms.ToTensor()
        self.frames = frames

    def _get_file_name(self, index):
        """
        Read consecutive frames within index-th data starting at the start-th frame
        :param index: number of video in dataset
        :return:
        """
        res = []
        start = np.random.randint(1, 8-self.frames)
        for i in range(start, start+self.frames):
            res.append(os.path.join(self.base_path, self.txt_file[index].strip(), 'im{}.png'.format(i)))
        return res

    @staticmethod
    def _get_random_sigma():
        r = np.random.rand()
        return 10 ** (1.5*r - 2)

    def _get_crop_h_w(self):
        h = np.random.randint(0, 256 - self.im_size + 1)
        w = np.random.randint(0, 448 - self.im_size + 1)
        return h, w

    def __getitem__(self, index):
        img_files = self._get_file_name(index)

        if not self.im_size is None:
            hs, ws = self._get_crop_h_w()
            gt = torch.zeros(3, self.im_size, self.im_size)
            noised = torch.zeros(self.frames+1, 3, self.im_size, self.im_size)
            sigma = self._get_random_sigma()
            for i, file in enumerate(img_files):
                img = Image.open(file)
                img = self.trans(img)[:, hs:hs+self.im_size, ws:ws+self.im_size]
                if i == self.frames//2:
                    gt = img
                noised[i, ...] = torch.clamp(img + sigma*torch.randn_like(img), 0.0, 1.0)
            noised[-1, ...] = sigma * torch.ones_like(gt)
        else:
            sigma = self._get_random_sigma()
            noised = []
            for i, file in enumerate(img_files):
                img = Image.open(file)
                img = self.trans(img)
                if i == self.frames//2:
                    gt = img
                noised.append(torch.clamp(img + sigma*torch.randn_like(img), 0.0, 1.0))
            noised.append(sigma*torch.ones_like(gt))
            noised = torch.stack(noised, dim=0)
        return noised, gt

    def __len__(self):
        return len(self.txt_file)


if __name__ == '__main__':
    dataset = Video_Provider(
        'H:/vimeo_septuplet/sequences',
        'H:/vimeo_septuplet/sep_trainlist.txt',
        im_size=256
    )
    tran = transforms.ToPILImage()
    for index, (data, gt) in enumerate(dataset):
        # for i in range(6):
        #     tran(data[i, ...]).save('{}_noisy_{}.png'.format(index, i), quality=100)
        # tran(gt).save('{}_gt.png'.format(index), quality=100)
        print(index)