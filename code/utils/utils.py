#
# utils.py
#
# This file is part of TSCNN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import sys
import torch
import numpy

from skimage.metrics import structural_similarity
from skimage.color import rgb2ycbcr
from importlib import import_module

from argument.argument import args


# RGGB 4通道HR转为3通道HR，即G=(G1+G2)/2
def RGGB2RGB(img):
    rggb = numpy.zeros([img.shape[0], img.shape[1], 3], dtype=float)
    rggb[:, :, 0] = img[:, :, 0]
    rggb[:, :, 1] = (img[:, :, 1] + img[:, :, 2])/2
    rggb[:, :, 2] = img[:, :, 3]
    return rggb


def psnr(input, target, rgb_range):
    r_input, g_input, b_input = input.split(1, 1)
    if target.shape[1] == 3:
        r_target, g_target, b_target = target.split(1, 1)
    if target.shape[1] == 4:
        r_target, g_target1, g_target2, b_target = target.split(1, 1)
        g_target = (g_target1 + g_target2)/2

    mse_r = (r_input - r_target).pow(2).mean()
    mse_g = (g_input - g_target).pow(2).mean()
    mse_b = (b_input - b_target).pow(2).mean()

    cpsnr = 10 * (rgb_range * rgb_range / ((mse_r + mse_g + mse_b) / 3)).log10()

    psnr = torch.tensor([[10 * (rgb_range * rgb_range / mse_r).log10(),
                         10 * (rgb_range * rgb_range / mse_g).log10(),
                         10 * (rgb_range * rgb_range / mse_b).log10(),
                         cpsnr]]).float()

    return psnr


def ssim(input, target, rgb_range):
    y1 = rgb2ycbcr(input)[:, :, 0]
    if target.shape[2] == 3:
        y2 = rgb2ycbcr(target)[:, :, 0]
    if target.shape[2] == 4:

        y2 = rgb2ycbcr(RGGB2RGB(target))[:, :, 0]

    c_s = structural_similarity(y1, y2, data_range=rgb_range, sigma=1.5, gaussian_weights=True,
                                use_sample_covariance=False)

    return torch.tensor([[c_s]]).float()


def import_fun(fun_dir, module):
    fun = module.split('.')
    m = import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def catch_exception(exception):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print('{}: {}.'.format(exc_type, exception), exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)


