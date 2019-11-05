#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net res101 --dataset pascal_voc_water --dataset_t water --save_dir $2