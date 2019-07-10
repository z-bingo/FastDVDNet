import torch
import torch.nn as nn
from arch.M_U_Net import M_U_Net


class FastDVDNet(nn.Module):
    def __init__(self, in_frames=5, color=True, sigma_map=True):
        """
        class initial
        :param in_frames: T-2, T-1, T, T+1, T+2, generally 5 frames
        :param color: now only color images are supported
        :param sigma_map: noise map, whose value is the estimation of noise standard variation
        """
        super(FastDVDNet, self).__init__()
        self.in_frames = in_frames
        channel = 3 if color else 1
        in1 = (3 + (1 if sigma_map else 0)) * channel
        in2 = 3 * channel
        self.block1 = M_U_Net(in1, channel)
        self.block2 = M_U_Net(in2, channel)

    def forward(self, input):
        """
        forward function
        :param input: [b, N, c, h, w], the concatenation of noisy frames and noise map
        :return: the noised frame corresponding to reference frame
        """
        # split the noisy frames and noise map
        frames, map = torch.split(input, self.in_frames, dim=1)
        b, N, c, h, w = frames.size()
        data_temp = []
        # first stage
        for i in range(self.in_frames-2):
            data_temp.append(self.block1(
                torch.cat([frames[:, i:i+3, ...].view(b, -1, h, w), map.squeeze(1)], dim=1),
                frames[:, i+1, ...]
            ))
        # second stage
        data_temp = torch.cat(data_temp, dim=1)
        return self.block2(data_temp, frames[:, N//2+N%2, ...])
