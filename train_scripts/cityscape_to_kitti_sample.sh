#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net vgg16 --dataset cityscape_car --dataset_t kitti_car --save_dir $2