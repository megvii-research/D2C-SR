'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
import pickle
from model.model import *
import megengine
import megengine.data as data
import megengine.distributed as dist
from config.config_test import args
from dataset.dataset import *
from metrics.PSNR import *
from metrics.SSIM import *
from tqdm import tqdm


logging = megengine.logger.get_logger()


def main():

    ngpus_per_node = 1
    if args.ngpus:
        ngpus_per_node = args.ngpus

    # launch processes
    test = dist.launcher(worker) if ngpus_per_node > 1 else worker
    test(args)


def worker(args):
    world_size = dist.get_world_size()

    # create dataset
    valid_dataloader, len_val = create_dataset(args)

    # create model
    model_D = D_Net(args)
    model_C = C_Net()


    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    print("model path: %s"%(args.checkpoint))
    model_D.load_state_dict(state["state_dict_D"])
    model_C.load_state_dict(state["state_dict_C"])

    def valid_step(image, label):
        out11, out12, out21, out22 = model_D(image)
        sr = model_C(out11, out12, out21, out22)
        psnr_it = cal_psnr(sr, label, args.scale)
        ssim_it = ssim(sr, label)
        if world_size > 1:
            psnr_it = F.distributed.all_reduce_sum(psnr_it) / world_size
            ssim_it = F.distributed.all_reduce_sum(ssim_it) / world_size
        return psnr_it, ssim_it.item()

    model_D.eval()
    model_C.eval()
    psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)
    logging.info(
        "PSNR [\033[1;31m{:.2f}\033[0m]  SSIM [\033[1;31m{:.3f}\033[0m] ".format(psnr_v, ssim_v))


def valid(func, data_queue, len_val):
    psnr_v = 0.
    ssim_v = 0.
    for step, (image, label) in enumerate(tqdm(data_queue)):
        image = megengine.tensor(image)
        label = megengine.tensor(label)
        psnr_it, ssim_it = func(image, label)
        psnr_v += psnr_it
        ssim_v += ssim_it
    test_num = step + 1
    psnr_v /= test_num
    ssim_v /= test_num
    assert test_num == len_val
    return psnr_v, ssim_v



def create_dataset(args):
    val_list_path = args.val_list_path
    test_list = open(val_list_path, 'r').readlines()
    len_val = len(test_list)
    valid_dataset = TestDataset(test_list)

    valid_sampler = data.SequentialSampler(valid_dataset, batch_size=1, drop_last=False)
    valid_dataloader = data.DataLoader(valid_dataset, sampler=valid_sampler, num_workers=args.workers)
    return valid_dataloader,len_val


if __name__ == "__main__":
    main()
