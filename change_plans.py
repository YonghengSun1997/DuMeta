# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Project    : nnUNet
# File       : change_plans.py
# Time       ：2022/10/24 21:36
# Author     ：Yongheng Sun
"""
import pickle
PATH = r'.\nnUNetPlansv2.1_plans_3D_IBIS6M.pkl'
your_PATH = r''

f = open(your_PATH,'rb')
your_data = pickle.load(f)

f2 = open(PATH,'rb')
data2 = pickle.load(f2)


your_data['plans_per_stage'] = data2['plans_per_stage']
f = open(your_PATH, 'wb')
pickle.dump(your_data, f)
f.close()
