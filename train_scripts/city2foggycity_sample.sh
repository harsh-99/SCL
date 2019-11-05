#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --save_dir $2