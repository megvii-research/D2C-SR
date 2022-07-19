'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

------------------------------------------------------------------------------
Part of the following code in this file refs to https://github.com/yulunzhang/RCAN
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.
--------------------------------------------------------------------------------

'''


import os, sys
sys.path.append(os.path.abspath('.'))
import megengine.module as nn
import model.model_utils as model_utils
import megengine as mge
import megengine.functional as F

class D_Net(nn.Module):
    def __init__(self, args, n_resblocks=4, n_feats=64,kernel_size=3,rgb_range = 1,
                 n_colors = 3,reduction = 16,act = nn.ReLU(),conv=model_utils.default_conv):
        super(D_Net, self).__init__()
        scale = args.scale
        self.sub_mean = model_utils.MeanShift(rgb_range)
        self.add_mean = model_utils.MeanShift(rgb_range, sign=1)

        m_head = [conv(n_colors, n_feats, kernel_size)]

        m_body1 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body1.append(conv(n_feats, n_feats, kernel_size))

        m_body11 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body11.append(conv(n_feats, n_feats, kernel_size))

        m_body12 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body12.append(conv(n_feats, n_feats, kernel_size))

        m_body2 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        m_body21 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body21.append(conv(n_feats, n_feats, kernel_size))

        m_body22 = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(2)
        ]
        m_body22.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail11 = [
            model_utils.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail12 = [
            model_utils.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail21 = [
            model_utils.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]
        m_tail22 = [
            model_utils.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.res_conv1 = conv(n_feats, n_feats, kernel_size)
        self.res_conv2 = conv(n_feats, n_feats, kernel_size)
        self.res_conv3 = conv(n_feats, n_feats, kernel_size)
        self.res_conv4 = conv(n_feats, n_feats, kernel_size)
        self.res_conv5 = conv(n_feats, n_feats, kernel_size)
        self.res_conv6 = conv(n_feats, n_feats, kernel_size)
        self.res_conv7 = conv(n_feats, n_feats, kernel_size)
        self.res_conv8 = conv(n_feats, n_feats, kernel_size)

        self.head = nn.Sequential(*m_head)

        self.body1 = nn.Sequential(*m_body1)
        self.body11 = nn.Sequential(*m_body11)
        self.body12 = nn.Sequential(*m_body12)

        self.body2 = nn.Sequential(*m_body2)
        self.body21 = nn.Sequential(*m_body21)
        self.body22 = nn.Sequential(*m_body22)

        self.tail11 = nn.Sequential(*m_tail11)
        self.tail12 = nn.Sequential(*m_tail12)
        self.tail21 = nn.Sequential(*m_tail21)
        self.tail22 = nn.Sequential(*m_tail22)

    def forward(self, x):
        x = self.sub_mean(x)
        x_head = self.head(x)

        res1 = self.body1(x_head)
        res2 = self.body2(x_head)

        res11 = self.body11(res1)
        res12 = self.body12(res1)

        res21 = self.body21(res2)
        res22 = self.body22(res2)

        res11=self.res_conv1(res11)
        res11 += res1
        res11=self.res_conv2(res11)
        res11 += x_head
        res11 = self.tail11(res11)
        res11 = self.add_mean(res11)

        res12=self.res_conv3(res12)
        res12 += res1
        res12=self.res_conv4(res12)
        res12 += x_head
        res12 = self.tail12(res12)
        res12 = self.add_mean(res12)

        res21=self.res_conv5(res21)
        res21 += res2
        res21=self.res_conv6(res21)
        res21 += x_head
        res21 = self.tail21(res21)
        res21 = self.add_mean(res21)

        res22=self.res_conv7(res22)
        res22 += res2
        res22=self.res_conv8(res22)
        res22 += x_head
        res22 = self.tail22(res22)
        res22 = self.add_mean(res22)

        return res11, res12, res21, res22

class C_Net(nn.Module):
    def __init__(self):
        super(C_Net, self).__init__()
        self.fusion_final = nn.Sequential(
            model_utils.BasicConv(12, 64, 3, stride=1, padding=1, relu=True),
            model_utils.BasicConv(64, 64, 3, stride=1, padding=1, relu=True),
            model_utils.BasicConv(64, 4, 3, stride=1, padding=1, relu=True))

    def forward(self, x1,x2,x3,x4):
        cat_out=F.concat((x1,x2,x3,x4),axis=1)
        mask = self.fusion_final(cat_out)
        mask = F.softmax(mask,axis=1)
        mask1 = F.expand_dims(mask[:,0,...],axis=1)
        mask2 = F.expand_dims(mask[:, 1, ...], axis=1)
        mask3 = F.expand_dims(mask[:, 2, ...], axis=1)
        mask4 = F.expand_dims(mask[:, 3, ...], axis=1)
        fusion=x1*mask1 + x2*mask2 +x3*mask3+x4*mask4
        return fusion


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


if __name__ == '__main__':
    from config.config import args
    model_D=D_Net(args)
    print(model_D)
