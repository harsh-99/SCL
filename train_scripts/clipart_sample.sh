#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net res101 --dataset pascal_voc_0712 --dataset_t clipart --save_dir $2