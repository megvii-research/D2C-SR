# coding :utf-8
import argparse
import os

parser = argparse.ArgumentParser(description="D2C-SR")
# loss weight
parser.add_argument('--debug', type=bool, default=False,
                    help='if debug or not')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path of checkpoint')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--val_list_path', type=str, default=None,
                    help='test list path')
parser.add_argument('--workers', type=int, default=8)

parser.add_argument("--scale", type=int, default=4,
                    help="Scale factor")
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--ngpus', type=int, default=1,
                    help='number of GPUs')

args = parser.parse_args()
