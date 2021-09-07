#
# common.py
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2021-01-20     Hengxin.Li   the first version

import os
import torch
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.io import savemat
from torch.nn.functional import interpolate as F
from argument.argument import args
from utils import log, timer, utils
def RGGB2RGB(img):
    rggb = torch.zeros([img.shape[0], 3, img.shape[2], img.shape[3]], dtype=torch.float).cuda()
    rggb[:, 0, :, :] = img[:, 0, :, :]
    rggb[:, 1, :, :] = (img[:, 1, :, :] + img[:, 2, :, :])/2
    rggb[:, 2, :, :] = img[:, 3, :, :]
    return rggb


def test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            for batch_index, (data, x_dem, target) in enumerate(d):
                data, x_dem, target = data.to(device, non_blocking=True), x_dem.to(device, non_blocking=True), target.to(device, non_blocking=True)
                try:
                    timer_test.restart()
                    model_out = model(data, x_dem)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out if len(model_out) == 1 else model_out[0]
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)
                    if target.shape[1] == 4:
                        target = RGGB2RGB(target)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint16(out_label)

                    all_ssim = utils.ssim(out_data, out_label, args.n_rgb_range).to(device)
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    info_log.write('{}_{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                        d.dataset.name,
                        batch_index,
                        all_psnr[:, -1].item(),
                        all_psnr[:, 0].item(),
                        all_psnr[:, 1].item(),
                        all_psnr[:, 2].item(),
                        all_ssim.item(),
                    ))

                    if args.b_save_results:
                        path = os.path.join('./experiments', args.s_experiment_name, d.dataset.name,
                                            'result_' + str(batch_index) + '.bmp')
                        cv2.imwrite(path, cv2.cvtColor(model_out[0, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)

            info_log.write('{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
            ))

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim = im_ssim.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim, timer_test_elapsed_ticks


def mosaic_bayer(rgb, pattern):
    num = np.zeros(len(pattern), dtype=np.uint8)
    p = [i for (i, val) in enumerate(pattern) if ((val == 'r') + (val == 'R'))]
    num[p] = 0
    p = [i for (i, val) in enumerate(pattern) if ((val == 'g') + (val == 'G'))]
    num[p] = 1
    p = [i for (i, val) in enumerate(pattern) if ((val == 'b') + (val == 'B'))]
    num[p] = 2

    mosaic = np.zeros(rgb.shape, dtype=np.uint8)
    mask = np.zeros(rgb.shape, dtype=np.uint8)

    mask[0::2, 0::2, num[0]] = 1
    mask[0::2, 1::2, num[1]] = 1

    mask[1::2, 0::2, num[2]] = 1
    mask[1::2, 1::2, num[3]] = 1

    mosaic[:, :, 0] = rgb[:, :, 0] * mask[:, :, 0]
    mosaic[:, :, 1] = rgb[:, :, 1] * mask[:, :, 1]
    mosaic[:, :, 2] = rgb[:, :, 2] * mask[:, :, 2]

    return mosaic, mask
