#
# argument.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import argparse

parser = argparse.ArgumentParser(description='template of demosaick')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--b_cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='number of GPU')
parser.add_argument('--b_cudnn', type=bool, default=True,
                    help='use cudnn')
parser.add_argument('--n_seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0',
                    help='CUDA_VISIBLE_DEVICES')

# Model specifications
parser.add_argument('--s_model', '-m', default='TSCNN.RDSRN',
                    help='model name')
parser.add_argument('--b_save_all_models', default=True,
                    help='save all intermediate models')
parser.add_argument('--b_load_best', type=bool, default=False,
                    help='use best model for testing')

# Data specifications
parser.add_argument('--dir_dataset', type=str, default='../DATA_testx2',
                    help='dataset directory')
parser.add_argument('--n_patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--n_rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--data_pack', type=str, default='packet/packet',  # train/test
                    choices=('packet', 'bin', 'ori'),
                    help='make binary data')

# Evaluation specifications
parser.add_argument('--s_eval_dataset', default='mcm.Mcm+kodak.Kodak+Urban100.Urban100',
                    help='evaluation dataset')
parser.add_argument('--b_test_only', type=bool, default=True,
                    help='set this option to test the model')
parser.add_argument('--pre_train', type=str, default='TSCNN_Lx2.pth',
                    help='pre-trained model directory')

# Log specifications
parser.add_argument('--s_experiment_name', type=str, default='test',
                    help='file name to save')
parser.add_argument('--b_save_results', type=bool, default=False,
                    help='save output results')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

parser.add_argument('--n_denseblock', type=int, default=3,
                    help='number of denseblock')
parser.add_argument('--n_skipblocks', type=int, default=3,
                    help='number of residual groups')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
