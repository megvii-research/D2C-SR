# coding :utf-8
import argparse
import os

lr_init = 0.0001
epoch = 20000
decay_epoch = 2000
lr_decay_rate = 0.5
batch_size = 4
INPUT_SIZE = 96
SCALE = 4
N_GPUs = 1


save_dir = "./experiments/"
ex_id = "D2CSR"
ex_name = "%s_%sx" % (ex_id, SCALE)

parser = argparse.ArgumentParser(description="D2C-SR")

parser.add_argument('--debug', type=bool, default=False,
                    help='if debug or not')
parser.add_argument('--load_checkpoint', type=bool, default=False,
                    help='if load checkpoint or not')
parser.add_argument("--ex_id", type=str, default=ex_id,
                    help="exp id")
parser.add_argument("--print-freq", type=int, default=10,
                    help="print freq")
parser.add_argument("--val-freq", type=int, default=20,
                    help="print freq")
parser.add_argument('--save', type=str, default=os.path.join(save_dir, ex_name),
                    help='file name to save')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint load path')
parser.add_argument("--batch-size", type=int, default=batch_size,
                    help="Number of images a batch")
parser.add_argument('--lr', type=float, default=lr_init,
                    help='learning rate')
parser.add_argument('--decay_epoch', type=int, default=decay_epoch,
                    help='learning rate decay')
parser.add_argument('--gamma', type=float, default=lr_decay_rate,
                    help='learning rate decay factor for step decay')
parser.add_argument('--epochs', type=int, default=epoch,
                    help='number of epochs to train')
parser.add_argument('--res_scale', type=float, default=20,
                    help='residual scaling')
parser.add_argument('--L2_w', type=float, default=1,
                    help='L2 loss weight')
parser.add_argument('--margin_same', type=float, default=2e-5,
                    help='margin same father branch')
parser.add_argument('--margin_diff', type=float, default=5e-5,
                    help='margin diff father branch')
parser.add_argument('--diver_w', type=float, default=2e-5,
                    help='weight of alpha')
parser.add_argument('--vgg_w', type=float, default=0.,
                    help='weight of vgg loss')
parser.add_argument('--diver_start_epoch', type=int, default=200,
                    help='diver loss start')
parser.add_argument('--diver_every', type=int, default=1000,
                    help='use triplet loss epoch')
parser.add_argument('--diver_epochs', type=int, default=200,
                    help='number of epochs using triplet')
parser.add_argument('--diver_decay_epoch', type=int, default=2000,
                    help='epoch triplet decay')
parser.add_argument('--diver_decay_rate', type=int, default=0.5,
                    help='diver decay rate')
parser.add_argument('--train_list_path', type=str, default=None,
                    help='train data list path')
parser.add_argument('--val_list_path', type=str, default=None,
                    help='val list path')
parser.add_argument("--patch_size", type=int, default=INPUT_SIZE,
                    help="Size of training patch.")
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight-decay', type=float, default=1e-8,
                    help='weight decay')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument("--scale", type=int, default=SCALE,
                    help="Scale factor")
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--ngpus', type=int, default=N_GPUs,
                    help='number of GPUs')
parser.add_argument('--ex_name', type=str, default=ex_name,
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')

args = parser.parse_args()
