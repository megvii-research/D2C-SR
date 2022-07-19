'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
from megengine.data.dataset import Dataset
from tqdm import tqdm
import cv2
from dataset.dataset_utils import *


class TrainDataset(Dataset):
    def __init__(self, data_list, args):
        self.PATCH_WIDTH = args.patch_size
        self.PATCH_HEIGHT = args.patch_size
        self.data_list = data_list
        self.img_list = []
        self.scale = args.scale
        num = 0
        for pair in tqdm(self.data_list):
            pair = pair.strip("\n")
            img_input = cv2.imread(pair.split(" ")[0])
            img_gt = cv2.imread(pair.split(" ")[1])
            scale = int(img_gt.shape[0] / img_input.shape[0])
            assert scale == args.scale
            self.scale = scale
            self.img_list.append([img_input, img_gt])
            num += 1

        print("Load TrainDataset Complete | Scale: %s  Data Nums: %s" % (self.scale, num))

    def getPatch(self, img, x, y, w, h):
        patch = img[y: y + h, x:x + w, :]
        patch = patch.astype(np.float32)
        patch = patch / 255.
        return patch

    def expand_transpose(self, patch):
        patch = np.transpose(patch, [2, 0, 1])
        return patch

    def __getitem__(self, index):
        input_img = self.img_list[index][0]
        gt_img = self.img_list[index][1]
        HEIGHT_in, WIDTH_in, _ = input_img.shape
        HEIGHT_gt, WIDTH_gt, _ = gt_img.shape
        scale = self.scale
        in_h, in_w, _ = input_img.shape
        gt_img = gt_img[0:in_h * scale, 0:in_w * scale, :]
        x_in = random.randint(0, WIDTH_in - self.PATCH_WIDTH - 1)
        y_in = random.randint(0, HEIGHT_in - self.PATCH_HEIGHT - 1)
        x_gt = x_in * scale
        y_gt = y_in * scale

        input_patch = self.getPatch(input_img, x_in, y_in, self.PATCH_WIDTH, self.PATCH_HEIGHT)
        gt_patch = self.getPatch(gt_img, x_gt, y_gt, self.PATCH_WIDTH * scale, self.PATCH_HEIGHT * scale)

        input_patch, gt_patch = augment(input_patch, gt_patch)
        input_patch = self.expand_transpose(input_patch)
        gt_patch = self.expand_transpose(gt_patch)

        return input_patch, gt_patch

    def __len__(self):
        return len(self.img_list)


class TestDataset(Dataset):

    def __init__(self, val_list):
        self.val_list = val_list
        self.img_list = []
        self.scale = 1
        num = 0
        for pair in tqdm(self.val_list):
            pair = pair.strip("\n")
            path = pair.split(" ")[0]
            im_name = path.split("/")[-1]
            img_input = cv2.imread(pair.split(" ")[0])
            img_gt = cv2.imread(pair.split(" ")[1])
            scale = int(img_gt.shape[0] / img_input.shape[0])
            self.scale = scale
            self.img_list.append([img_input, img_gt, im_name])
            num += 1
        print("Load TestDataset Complete | Scale: %s  Data Nums: %s" % (self.scale, num))

    def __getitem__(self, index):
        input_img = self.img_list[index][0]
        gt_img = self.img_list[index][1]
        in_h, in_w, _ = input_img.shape
        scale = self.scale
        input_img = input_img.astype(np.float32)
        input_img = input_img / 255.
        input_patch = np.transpose(input_img, [2, 0, 1])
        gt_img = gt_img[0:in_h * scale, 0:in_w * scale, :]
        gt_img = gt_img.astype(np.float32)
        gt_img = gt_img / 255.
        gt_patch = np.transpose(gt_img, [2, 0, 1])

        return input_patch, gt_patch

    def __len__(self):
        return len(self.img_list)
