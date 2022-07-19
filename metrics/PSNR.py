'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os,sys
import math
sys.path.append(os.path.abspath('.'))
import megengine.functional as F
import megengine as mge


def cal_psnr(sr, hr, shave):
    if not sr.shape == hr.shape:
        raise ValueError('Input images must have the same dimensions.')
    sr = F.squeeze(sr,axis=0)
    hr = F.squeeze(hr, axis=0)
    sr = F.clip(sr, 0, 1)
    hr = F.clip(hr, 0, 1)
    diff = (sr - hr)
    gray_coeffs = [65.738, 129.057, 25.064]
    gray_coeffs_t = mge.Tensor(gray_coeffs,is_const=True)
    convert = gray_coeffs_t.reshape(3, 1, 1) / 256
    diff = F.mul(diff,convert)
    diff = F.sum(diff,axis=0,keepdims=True)
    valid = diff[..., shave:-shave, shave:-shave]
    mse = F.mean(F.pow(valid,2))
    return -10 * math.log10(mse)


if __name__ == '__main__':
    a = mge.random.uniform(0,1,(3,10,10))
    b = mge.random.uniform(0,1,(3,10,10))
    print(cal_psnr(a,b,4))


