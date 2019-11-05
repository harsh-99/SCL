#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net vgg16 --dataset kitti_car --dataset_t cityscape_car --save_dir $2