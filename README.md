# Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan
by [Yongheng Sun](https://yonghengsun1997.github.io/), [Fan Wang](https://gr.xjtu.edu.cn/en/web/fan.wang), Jun Shu, Haifeng Wang, [Li Wang](https://www.med.unc.edu/radiology/directory/li-wang/), [Deyu Meng](https://gr.xjtu.edu.cn/en/web/dymeng/1), [Chunfeng Lian](https://gr.xjtu.edu.cn/en/web/cflian). 

## Introduction

This repository is for our ICCV 2023 paper '[Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan](https://arxiv.org/abs/2308.06774)'. 


![](./picture/DuMeta.PNG)

## A. You can follow the steps below to pretrain and finetune your model on your data with DuMeta

## 1 Preparation

### 1.1 Install nnUNetv1.

### 1.2 Process your data as required by nnUNetv1.
```
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```
### 1.3 Run nnUNetv1 baseline on your data. Save your Generic_UNet, DataLoader3D, and get_moreDA_augmentation hyperparameters.
```
nnUNet_train 3d_fullres nnUNetTrainerV2 TASK_NAME_OR_ID FOLD (additional options)
```
### 1.4 Replace your nnUNet folder with this repository (I rewrote the files in the folders of \nnunet\network_architecture and \nnunet\training\network_training).

### 1.5 Import Generic_UNet, DataLoader3D, and get_moreDA_augmentation with hyperparameters in step 1.3, as in main.py.

## 2 Training and Testing

### 2.1 Meta-train (Pretrain).
```
python main.py
```

### 2.2 Meta-test (Finetune).

### Load model parameters of step 2.1, and run baseline as step 1.3.

## B. Or you can load our pretrained model and finutune on your own data.
1. You can download the pretrained model via https://drive.google.com/file/d/1t6nCM376LBVHXjktr52k8KeDwuZZTLy2/view?usp=drive_link.
2. Process your data as required by nnUNetv1 using below command:\
   ```
   nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
   ```
3. then change nnUNetPlansv2.1_plans_3D.pkl to change patch size of input:\
   ```
   python change_plans.py
   ```

4. finetune your model using below command:\
   ```
   nnUNet_train 3d_fullres nnUNetTrainerV2_FT TASK_NAME_OR_ID FOLD (additional options)
   ```

## Acknowledgement
The code is based on nnUNetv1 (https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Citation

If you find this project useful for your research, please consider citing:

```bibtex
@inproceedings{sun2023dual,
  title={Dual Meta-Learning with Longitudinally Consistent Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan},
  author={Sun, Yongheng and Wang, Fan and Shu, Jun and Wang, Haifeng and Wang, Li and Meng, Deyu and Lian, Chunfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21118--21128},
  year={2023}
}
```
