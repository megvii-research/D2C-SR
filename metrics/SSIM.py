'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

------------------------------------------------------------------------------
Part of the following code in this file refs to https://github.com/Po-Hsun-Su/pytorch-ssim
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.
--------------------------------------------------------------------------------

'''

import megengine as mge
import megengine.functional as F
import numpy as np
from math import exp
import megengine.module as nn
import pdb


def gaussian(window_size, sigma):
    gauss = mge.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)], is_const=True)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = F.expand_dims(gaussian(window_size, 1.5),axis=1)
    _1D_window = F.matmul(_1D_window,_1D_window,transpose_a=False,transpose_b=True)
    _1D_window = F.expand_dims(_1D_window,axis=0)
    _2D_window = F.expand_dims(_1D_window,axis=0)
    _2D_window = F.broadcast_to(_2D_window, (channel, 1, window_size, window_size))
    window = mge.Tensor(_2D_window)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = F.pow(mu1,2)
    mu2_sq = F.pow(mu2,2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            # if img1.is_cuda:
            #     window = window.cuda(img1.get_device())
            # window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):

    while len(img1.shape) < 4:
        img1 = F.expand_dims(img1, axis=0)
        img2 = F.expand_dims(img2, axis=0)

    gray_coeffs = [65.738, 129.057, 25.064]
    gray_coeffs_t = mge.Tensor(gray_coeffs, is_const=True)
    convert = gray_coeffs_t.reshape(1, 3, 1, 1) / 256
    img1 = F.mul(img1,convert)
    img1 = F.sum(img1,axis=1,keepdims=True)
    img2 = F.mul(img2,convert)
    img2 = F.sum(img2,axis=1,keepdims=True)
    (_,channel,_,_) = img1.shape
    window = create_window(window_size, channel)

    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    # window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == '__main__':
    a = mge.random.uniform(0,1,(3,10,10))
    b = mge.random.uniform(0,1,(3,10,10))
    print(ssim(a,b))
