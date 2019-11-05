# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths


import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss, CrossEntropy

from model.utils.parser_func_multi import parse_args, set_dataset_args

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16_SCL import vgg16
    from model.faster_rcnn.resnet_SCL import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, gc1 = args.gc1, gc2=args.gc2, gc3 = args.gc3)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            gc1 = args.gc1, gc2=args.gc2, gc3 = args.gc3)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    # print(fasterRCNN)
    # exit()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (args.load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size) # len(dataloader_s)#
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    count_iter = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            #eta = 1.0
            count_iter += 1
            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label,out_d_inst, out_d1, out_d2, out_d3 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()


            # domain label
            domain_s2 = domain_s3 = Variable(torch.zeros(out_d2.size(0)).long().cuda())
            domain_s_p = Variable(torch.zeros(out_d_inst.size(0)).long().cuda())
            # k=1th loss
            dloss_s1 = 0.5 * torch.mean(out_d1 ** 2)
            # k=2nd loss
            dloss_s2 = 0.5 * CrossEntropy(out_d2, domain_s2) * 0.15
            # k = 3rd loss 
            dloss_s3 = 0.5 * FL(out_d3, domain_s3)
            # instance alignment loss
            dloss_s_p = 0.5 * FL(out_d_inst, domain_s_p)

            #put target data into variable
            im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            #gt is empty
            # gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()
            out_d_inst, out_d1, out_d2, out_d3 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, target=True)
            # domain label
            domain_t2 = domain_t3 = Variable(torch.ones(out_d2.size(0)).long().cuda())
            domain_t_p = Variable(torch.ones(out_d_inst.size(0)).long().cuda())
            # k=1th loss
            dloss_t1 = 0.5 * torch.mean((1 - out_d1) ** 2)
            # k=2nd loss
            dloss_t2 = 0.5 * CrossEntropy(out_d2, domain_t2) * 0.15
            # k = 3rd loss 
            dloss_t3 = 0.5 * FL(out_d3, domain_t3)
            # instance alignment loss
            dloss_t_p = 0.5 * FL(out_d_inst, domain_t_p)
            if args.dataset == 'sim10k':
                loss += (dloss_s1 + dloss_t1 + dloss_s2 + dloss_t2 + dloss_s3 + dloss_t3 + dloss_s_p + dloss_t_p) * args.eta
                # print(args.eta)
            else:
                loss += (dloss_s1 + dloss_t1 + dloss_s2 + dloss_t2 + dloss_s3 + dloss_t3 + dloss_s_p + dloss_t_p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s1 = dloss_s1.item()
                    dloss_t1 = dloss_t1.item()
                    dloss_s2 = dloss_s2.item()
                    dloss_t2 = dloss_t2.item()
                    dloss_s3 = dloss_s3.item()
                    dloss_t3 = dloss_t3.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                # logger = open('./logger/logger_1_local_2_global_3_fwd_with_CE_retry6.txt', 'a')

                # logger.write("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                #       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                
                # logger.write("\nfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                
                # logger.write(
                  #   "\nrpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, dloss s1: %.4f, dloss t1: %.4f, dloss s2: %.4f, dloss t2: %.4f, dloss s3: %.4f, dloss t3: %.4f, dinst_s: %.4f, dinst_t: %.4f, eta: %.4f \n" \
                  #   % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s1, dloss_t1, dloss_s2, dloss_t2, dloss_s3, dloss_t3, dloss_s_p, dloss_t_p,
                  #      args.eta))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, dloss s1: %.4f, dloss t1: %.4f, dloss s2: %.4f, dloss t2: %.4f, dloss s3: %.4f, dloss t3: %.4f, dinst_s: %.4f, dinst_t: %.4f, eta: %.4f" \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, dloss_s1, dloss_t1, dloss_s2, dloss_t2, dloss_s3, dloss_t3, dloss_s_p, dloss_t_p,
                       args.eta))
                # logger.close()
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()
        save_name = os.path.join(output_dir,
                                 'globallocal_target_{}_eta_{}_gc1_{}_gc2_{}_gc3_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.gc1, args.gc2, args.gc3, args.gamma,
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
