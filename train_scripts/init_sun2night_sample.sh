#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net res101 --dataset init_sunny --dataset_t init_night --save_dir $2