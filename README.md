# Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan
by [Yongheng Sun](https://yonghengsun1997.github.io/), [Fan Wang](https://gr.xjtu.edu.cn/en/web/fan.wang), Jun Shu, Haifeng Wang, [Li Wang](https://www.med.unc.edu/radiology/directory/li-wang/), [Deyu Meng](https://gr.xjtu.edu.cn/en/web/dymeng/1), [Chunfeng Lian](https://gr.xjtu.edu.cn/en/web/cflian). 

## Introduction

This repository is for our ICCV 2023 paper '[Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan](https://arxiv.org/abs/2308.06774)'. 


![](./picture/DuMeta.PNG)


## You can follow the steps below to train your model on your data with DuMeta

## 1 Preparation

### 1.1 Install nnUNetv1.

### 1.2 Process your data as required by nnUNetv1.

### 1.3 Run nnUNetv1 baseline on your data. Save your Generic_UNet, DataLoader3D, and get_moreDA_augmentation hyperparameters.

### 1.4 Replace your nnUNet folder with this repository (I rewrote the files in the folders of \nnunet\network_architecture and \nnunet\training\network_training).

### 1.5 Import Generic_UNet, DataLoader3D, and get_moreDA_augmentation with hyperparameters in step 1.3, as in main.py.

## 2 Training and Testing

### 2.1 Meta-train (Pretrain).
```
python main.py
```

### 2.2 Meta-test (Finetune).

### Load model parameters of step 2.1, and run baseline as step 1.3.

## Acknowledgement
The code is based on nnUNetv1 (https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{sun2023dual,
  title={Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan},
  author={Sun, Yongheng and Wang, Fan and Shu, Jun and Wang, Haifeng and Meng, Li Wang and Lian, Chunfeng and others},
  journal={arXiv preprint arXiv:2308.06774},
  year={2023}
}
```
