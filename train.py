'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
import time
import pickle
import numpy as np
from model.model import *
import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.distributed as dist
import megengine.optimizer as optim
from dataset.dataset import *
from metrics.PSNR import *
from metrics.SSIM import *
from loss.divergence import trip_loss

from config.config import args

logging = megengine.logger.get_logger()


def main():
    ngpus_per_node = 1
    if args.ngpus:
        ngpus_per_node = args.ngpus

    # launch processes
    train = dist.launcher(worker) if ngpus_per_node > 1 else worker
    train(args)


def worker(args):
    diver_w = 0.
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("world size : %s" % world_size)
    if rank == 0:
        os.makedirs(os.path.join(args.save), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, "log.txt"))

    # create dataset
    train_dataloader, valid_dataloader, len_train, len_val = create_dataset(args)
    steps_per_epoch = len_train // args.batch_size
    print("steps_per_epoch : %s " % steps_per_epoch)
    train_queue = iter(train_dataloader)

    # create model
    model_D = D_Net(args)
    model_C = C_Net()

    if (args.load_checkpoint):
        with open(args.checkpoint, "rb") as f:
            state = pickle.load(f)
        print("model path: %s  |  PSNR: [%s]" % (args.checkpoint, state["psnr"]))
        model_D.load_state_dict(state["state_dict_D"])
        model_C.load_state_dict(state["state_dict_C"])

    # Sync parameters
    if world_size > 1:
        dist.bcast_list_(model_D.parameters(), dist.WORLD)
    if world_size > 1:
        dist.bcast_list_(model_C.parameters(), dist.WORLD)

    # Autodiff gradient manager
    gm_D = autodiff.GradManager().attach(
        model_D.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None)
    gm_C = autodiff.GradManager().attach(
        model_C.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None)

    # define Optimizer
    opt_D = optim.Adam(model_D.parameters(), lr=args.lr, weight_decay=args.weight_decay * world_size,
                       # scale weight decay in "SUM" mode
                       )
    opt_C = optim.Adam(model_C.parameters(), lr=args.lr, weight_decay=args.weight_decay * world_size,
                       # scale weight decay in "SUM" mode
                       )

    # train and valid
    def train_step_D(image, label):
        with gm_D:
            out11, out12, out21, out22 = model_D(image)
            loss11 = F.nn.square_loss(out11, label)
            loss12 = F.nn.square_loss(out12, label)
            loss21 = F.nn.square_loss(out21, label)
            loss22 = F.nn.square_loss(out22, label)
            loss_D = loss11 + loss12 + loss21 + loss22

            if (diver_w > 0.):
                loss_trip = trip_loss(out11, out12, out21, out22, label, diver_w, args.margin_same, args.margin_diff)
                loss_trip = loss_trip.item()
            else:
                loss_trip = 0.

            loss_divergence = loss_D + loss_trip

            gm_D.backward(loss_divergence)
            opt_D.step().clear_grad()
        out11, out12, out21, out22 = model_D(image)
        return loss_D, loss_trip, loss_divergence, out11, out12, out21, out22

    def train_step_C(out11, out12, out21, out22, label):
        with gm_C:
            fusion = model_C(out11, out12, out21, out22)
            loss_C = F.nn.square_loss(fusion, label)
            gm_C.backward(loss_C)
            opt_C.step().clear_grad()

        return loss_C

    def valid_step(image, label):
        out11, out12, out21, out22 = model_D(image)
        sr = model_C(out11, out12, out21, out22)
        psnr_it = cal_psnr(sr, label, args.scale)
        ssim_it = ssim(sr, label)
        if world_size > 1:
            psnr_it = F.distributed.all_reduce_sum(psnr_it) / world_size
            ssim_it = F.distributed.all_reduce_sum(ssim_it) / world_size
        return psnr_it, ssim_it

    # multi-step learning rate scheduler with warmup
    def adjust_learning_rate(step, opt):
        lr = args.lr * (args.gamma ** ((step / steps_per_epoch) // args.decay_epoch))
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    def adjust_diver(epoch, diver_init, diver_start_epoch, diver_every, diver_epochs, diver_decay_epoch,
                     diver_decay_rate):
        if (epoch <= diver_start_epoch):
            return 0.
        else:
            diver_w = diver_init * (diver_decay_rate ** (epoch // diver_decay_epoch))
            if (epoch % diver_every < diver_epochs):
                return diver_w
            else:
                return 0.

    # start training
    for step in range(0, int(args.epochs * steps_per_epoch)):

        lr_C = adjust_learning_rate(step, opt_C)
        lr_D = adjust_learning_rate(step, opt_D)
        diver_w = adjust_diver(step // steps_per_epoch, args.diver_w, args.diver_start_epoch, args.diver_every,
                               args.diver_epochs, args.diver_decay_epoch, args.diver_decay_rate)
        t_step = time.time()

        image, label = next(train_queue)
        image = megengine.tensor(image)
        label = megengine.tensor(label)
        t_data = time.time() - t_step
        loss_D, loss_trip, loss_divergence, out11, out12, out21, out22 = train_step_D(image, label)
        loss_C = train_step_C(out11, out12, out21, out22, label)
        t_train = time.time() - t_step
        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "[{}]  Epoch {} Step {} Loss_D={:.5} Loss_trip={:.5} Loss_C={:.5}  lr={:.5}  times={:.2}s".format(
                    args.ex_name,
                    step // steps_per_epoch,
                    step,
                    loss_D.item(),
                    loss_trip,
                    loss_C.item(),
                    lr_D, t_train
                    ))
        if ((step + 1) % (steps_per_epoch * args.val_freq) == 0):
            model_D.eval()
            model_C.eval()
            psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)
            model_D.train()
            model_C.train()
            megengine.save(
                {
                    "state_dict_D": model_D.state_dict(),
                    "state_dict_C": model_C.state_dict(),
                },
                os.path.join(args.save, "checkpoint.pkl"),
            ) if rank == 0 else None
            logging.info(
                "[{}]  PSNR [\033[1;31m{:.5f}\033[0m]  SSIM [\033[1;31m{:.5f}\033[0m]".format(
                    args.ex_name, psnr_v, ssim_v))


def valid(func, data_queue, len_val):
    psnr_v = 0.
    ssim_v = 0.
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image)
        label = megengine.tensor(label)
        psnr_it, ssim_it = func(image, label)
        psnr_v += psnr_it
        ssim_v += ssim_it
    test_num = step + 1
    psnr_v /= test_num
    ssim_v /= test_num
    assert test_num == len_val
    return psnr_v, ssim_v.item()


def create_dataset(args):
    train_list_path = args.train_list_path
    val_list_path = args.val_list_path
    if args.debug:
        train_list = open(train_list_path, 'r').readlines()[:args.batch_size]
        test_list = open(val_list_path, 'r').readlines()[:1]
    else:
        train_list = open(train_list_path, 'r').readlines()
        test_list = open(val_list_path, 'r').readlines()

    assert not args.batch_size // args.ngpus == 0

    len_train = len(train_list)
    len_val = len(test_list)
    train_dataset = TrainDataset(train_list, args)
    valid_dataset = TestDataset(test_list)

    train_sampler = data.Infinite(
        data.RandomSampler(train_dataset, batch_size=args.batch_size // args.ngpus, drop_last=True))
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, num_workers=args.workers)
    valid_sampler = data.SequentialSampler(valid_dataset, batch_size=1, drop_last=False)
    valid_dataloader = data.DataLoader(valid_dataset, sampler=valid_sampler, num_workers=args.workers)
    return train_dataloader, valid_dataloader, len_train, len_val


if __name__ == "__main__":
    main()
