# TSCNN

Kan Chang, Hengxin Li, Yufei Tan, Pak Lun Kevin Ding, Baoxin Li, " A Two-Stage Convolutional Neural Network for Joint Demosaicking and Super-Resolution", Submitted to IEEE Trans. Circuits Syst. Video Technol. 2021.

## Dependencies
* Python == 3.6
* Pytorch == 1.6.0
* torchvision == 0.7.0
* numpy == 1.18.5
* skimage == 0.16.2

## Contents
1. [Abstract](#abstract)
2. [Architecture](#architecture)
3. [Train](#train)
4. [Test](#test)
5. [Note](#note)
6. [Results](#results)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

## Abstract
As two practical and important image processing tasks, color demosaicking (CDM) and super-resolution (SR) have been studied for decades. However, most literature studies these two tasks independently, ignoring the potential benefits of a joint solution. In this paper, aiming at efficient and effective joint demosaicking and super-resolution (JDSR), a well-designed two-stage convolutional neural network (CNN) architecture is proposed. For the first stage, by making use of the sampling-pattern information, a pattern-aware feature extraction (PFE) module extracts features directly from the Bayer-sampled low-resolution (LR) image, while keeping the resolution of the extracted features the same as the input. For the second stage, a dual-branch feature refinement (DFR) module effectively decomposes the features into two components with different spatial frequencies, on which different learning strategies are applied. On each branch of the DFR module, the feature refinement unit, namely, densely-connected dual-path enhancement blocks (DDEB), establishes a sophisticated nonlinear mapping from the LR space to the high-resolution (HR) space. To achieve strong representational power, two paths of transformations and the channel attention mechanism are adopted in DDEB. Extensive experiments demonstrate that the proposed method is superior to the sequential combination of state-of-the-art (SOTA) CDM and SR methods. Moreover, with much smaller model size, our approach also surpasses other SOTA JDSR methods.

## Architecture

![TSCNN](/Figs/TSCNN.png)
The overall structure of the proposed TSCNN.

## Data

Training data [DIV2K (800 training + 100 validtion images)]

Benchmark data [(McM, Kodak)]

## Train

The source code for training our TSCNN will be available after the publication of the paper.

## Test

1. We provide six pre-trained models in the `pretrained_models/`.

2. We provide dataloader for McM and Kodak datasets, therefore you can evaluate the model on these datasets.

3. Cd to './code/argument', run the following scripts to test models.

    ```bash
    # Scale 2,3,4
    #-------------TSCNN_L_x2 
    python main.py --s_model TSCNN.RDSRN  --RDNconfig A --n_patch_size 96   --dir_dataset DATA_testx2 --pre_train TSCNN_Lx2.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak
    #-------------TSCNN_L_x3 
    python main.py --s_model TSCNN.RDSRN  --RDNconfig A --n_patch_size 144  --dir_dataset DATA_testx3 --pre_train TSCNN_Lx3.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak
    #-------------TSCNN_L_x4 
    python main.py --s_model TSCNN.RDSRN  --RDNconfig A --n_patch_size 192  --dir_dataset DATA_testx4 --pre_train TSCNN_Lx4.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak
    #-------------TSCNN_H_x2 
    python main.py --s_model TSCNN.RDSRN  --RDNconfig B --n_patch_size 96   --dir_dataset DATA_testx2 --pre_train TSCNN_Hx2.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak
    #-------------TSCNN_H_x3
    python main.py --s_model TSCNN.RDSRN  --RDNconfig B --n_patch_size 144  --dir_dataset DATA_testx3 --pre_train TSCNN_Hx3.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak
    #-------------TSCNN_H_x4
    python main.py --s_model TSCNN.RDSRN  --RDNconfig B --n_patch_size 192  --dir_dataset DATA_testx4 --pre_train TSCNN_Hx4.pth --b_test_only --s_eval_dataset mcm.Mcm+kodak.Kodak

    ```


## Note
### Prepare beachmark data
* To improve the speed of reading pictures, the data is pickled into a binary file(.bin). The pickle data contains mosaicked LR images(data), HR(label), demosaicked LR image by MLRI(x_dem).
* To generate the mosaicked LR images, we first use bicubic interpolation on the original images to obtain the LR images and then apply Bayer sampling on the LR images.
* The mosaicked LR image is also demosaicked in parallel by MLRI.
### Evaluate the results
* If you want to save the output images for each dataset, you need to add `--b_save_results True` to test commands.

## Results
### Quantitative Results
![PSNR_SSIM_Lightweight](/Figs/psnr_ssim.png)
Quantitative comparison among different models on five datasets (Average CPSNR (dB) /SSIM). The numbers in red and blue indicate the best and the second-best methods, respectively.

### Visual Results

![Visual_Result](/Figs/image92_Urban100_x2.bmp)
Results of img92 from Urban100 (×2). From left to right and top to bottom: DJDD + RCAN, CDMCNN + RCAN, 3-Stage + RCAN, TENet, RDSEN, TSCNN-L, TSCNN-H, Original image.

![Visual_Result](/Figs/image60_Urban100_x3.bmp)
Results of img60 from Urban100 (×3). From left to right and top to bottom: DJDD + RCAN, CDMCNN + RCAN, 3-Stage + RCAN, TENet, RDSEN, TSCNN-L, TSCNN-H, Original image.

## Citation
The citation information will be available after the publication of the paper.

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). 
We also refer to some other work such as [RCAN](https://github.com/yulunzhang/RCAN), [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter), [partial conv](https://github.com/NVIDIA/partialconv).
We thank these authors for sharing their codes.

