#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python test_net_SCL.py --cuda --net vgg16 --dataset kitti_car --load_name $2