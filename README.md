# Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan
by [Yongheng Sun](https://yonghengsun1997.github.io/), [Fan Wang](https://gr.xjtu.edu.cn/en/web/fan.wang), Jun Shu, Haifeng Wang, [Li Wang](https://www.med.unc.edu/radiology/directory/li-wang/), [Deyu Meng](https://gr.xjtu.edu.cn/en/web/dymeng/1), [Chunfeng Lian](https://gr.xjtu.edu.cn/en/web/cflian). 

## Introduction

This repository is for our ICCV 2023 paper '[Dual Meta-Learning with Longitudinally Generalized Regularization for One-Shot Brain Tissue Segmentation Across the Human Lifespan](https://arxiv.org/abs/2308.06774)'. 


![](./picture/DuMeta.PNG)

## You can follow the steps below to pretrain and finetune your model on your data with DuMeta

## 1 Preparation

### 1.1 Install nnUNetv1.

### 1.2 Process your data as required by nnUNetv1.
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity

### 1.3 Run nnUNetv1 baseline on your data. Save your Generic_UNet, DataLoader3D, and get_moreDA_augmentation hyperparameters.
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD (additional options)

### 1.4 Replace your nnUNet folder with this repository (I rewrote the files in the folders of \nnunet\network_architecture and \nnunet\training\network_training).

### 1.5 Import Generic_UNet, DataLoader3D, and get_moreDA_augmentation with hyperparameters in step 1.3, as in main.py.

## 2 Training and Testing

### 2.1 Meta-train (Pretrain).
```
python main.py
```

### 2.2 Meta-test (Finetune).

### Load model parameters of step 2.1, and run baseline as step 1.3.

## Or you can load our pretrained model and finutune on your own data.
1. You can download the pretrained model via https://drive.google.com/file/d/1t6nCM376LBVHXjktr52k8KeDwuZZTLy2/view?usp=drive_link.
2. Process your data as required by nnUNetv1 using below command:\
   &emsp;nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
3. then change nnUNetPlansv2.1_plans_3D.pkl to change patch size of input:\
    &emsp python change_plans.py
4. Replace nnUNetTrainerV2.initialize of nnUNetTrainerV2.py as follows:\
def initialize_network(self):\
    &emsp; num_input_channels = 1\
    &emsp; base_num_features = 32\
    &emsp; num_classes = 4\
    &emsp; conv_per_stage = 2\
    &emsp; conv_op = nn.Conv3d\
    &emsp; dropout_op = nn.Dropout3d\
    &emsp; norm_op = nn.InstanceNorm3d\
    &emsp; norm_op_kwargs = {'eps': 1e-5, 'affine': True}\
    &emsp; dropout_op_kwargs = {'p': 0, 'inplace': True}\
    &emsp; net_nonlin = nn.ReLU\
    &emsp; net_nonlin_kwargs = {'inplace': True}\
    &emsp; net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]\
    &emsp; net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]\
    &emsp; network = Generic_UNet(num_input_channels, base_num_features, num_classes,
                           5,
                           conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                           dropout_op_kwargs,
                           net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                           net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)\
    &emsp; if torch.cuda.is_available():\
        &emsp; &emsp; network.cuda()\
    &emsp; network.inference_apply_nonlin = softmax_helper\
    &emsp; network.load_state_dict(torch.load('./checkpoint/checkpoints.pth'))\
              &emsp;     for i, param in enumerate(network.parameters()):\
               &emsp;&emsp;     if (i > 87) or (i < 40):  # 前面一些参数冻结\
                     &emsp;&emsp;&emsp;   param.requires_grad = True
6. finetune your model using below command:\
   nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD (additional options)


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
