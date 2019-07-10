import torch
import torch.nn as nn
from arch.M_U_Net import M_U_Net

"""
The class if the implement of single-stage FastDVDNet
"""
class Single_Stage(nn.Module):
    def __init__(self, in_frames=5, color=True, sigma_map=True):
        super(Single_Stage, self).__init__()
        self.in_frames = in_frames
        channel = 3 if color else 1
        in1 = (3 + (1 if sigma_map else 0)) * channel
        self.block = M_U_Net(in1, channel)

    def forward(self, data):
        frames, map = torch.split(data, self.in_frames, dim=1)
        b, N, c, h, w = frames.size()
        return self.block1(
            torch.cat([frames.view(b, -1, h, w), map.squeeze(1)], dim=1),
            frames[:, N//2+N%2, ...]
        )
