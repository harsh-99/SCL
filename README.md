# Pytorch implementation of [SCL-Domain-Adaptive-Object-Detection](https://arxiv.org/abs/1911.02559) 
## Introduction 
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) repository to setup the environment. This code is based on the implemenatation of **Strong-Weak Distribution Alignment for Adaptive Object Detection**. We used Pytorch 0.4.0 for this project. The different version of pytorch will cause some errors, which have to be handled based on each envirionment.
<br />
For convenience, this repository contains implementation of: <br />
* SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses ([link](http://arxiv.org/abs/1911.02559))<br />
* Strong-Weak Distribution Alignment for Adaptive Object Detection, CVPR'19 ([link](https://arxiv.org/pdf/1812.04798.pdf)) <br />
* Domain Adaptive Faster R-CNN for Object Detection in the Wild, CVPR'18 (Our re-implementation) ([link](https://arxiv.org/pdf/1803.03243.pdf)) <br />

### Data preparation <br />
We have included the following set of datasets for our implementation: <br />
* **CitysScapes, FoggyCityscapes**: Download website [Cityscapes](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data) <br />
* **Clipart, WaterColor**: Dataset preparation instruction link [Cross Domain Detection](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets). <br />
* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. <br />
* **Sim10k**: Website [Sim10k](https://fcav.engin.umich.edu/sim-dataset/) <br />
* **Cityscapes-Translated Sim10k**: TBA <br />
* **KITTI** - For data prepration please follow [VOD-converter](https://github.com/umautobots/vod-converter) <br />
* **INIT** - Download the dataset from this [website](http://zhiqiangshen.com/projects/INIT/index.html) and data preparation file can be found in this repository in [data preparation folder](https://github.com/harsh-99/SCL-Domain-adaptive-object-detection/tree/master/lib/datasets/data_prep).

It is important to note that we have written all the codes for Pascal VOC format. For example the dataset cityscape is stored as: <br />

```
$ cd cityscape/VOC2012 
$ ls
Annotations  ImageSets  JPEGImages
$ cd ImageSets/Main
$ ls
train.txt val.txt trainval.txt test.txt
```
**Note:** If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in *lib/datasets/*, and add it to *lib/datasets/factory.py*, *lib/datasets/config_dataset.py*. Then, add the dataset option to *lib/model/utils/parser_func.py* and *lib/model/utils/parser_func_multi.py*.

### Data path <br />
Write your dataset directories' paths in lib/datasets/config_dataset.py.

for example 
```
__D.CLIPART = "./clipart"
__D.WATER = "./watercolor"
__D.SIM10K = "Sim10k/VOC2012"
__D.SIM10K_CYCLE = "Sim10k_cycle/VOC2012"
__D.CITYSCAPE_CAR = "./cityscape/VOC2007"
__D.CITYSCAPE = "../DA_Detection/cityscape/VOC2007"
__D.FOGGYCITY = "../DA_Detection/foggy/VOC2007"

__D.INIT_SUNNY = "./init_sunny"
__D.INIT_NIGHT = "./init_night"
```
### Pre-trained model <br/>

We used two pre-trained models on ImageNet as backbone for our experiments, VGG16 and ResNet101. You can download these two models from:

* VGG16 - [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth) <br />
* Resnet101 - [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)<br />

To provide their path in the code check __C.VGG_PATH and __C.RESNET_PATH at lib/model/utils/config.py.
<br />

**Our trained model** <br />
We are providing our models for foggycityscapes, watercolor and clipart.<br />
1) Adaptation form cityscapes to foggycityscapes:<br />
* VGG16 - [Google Drive](https://drive.google.com/open?id=1p5huR5TI_GLHCOwNXdZ4-BMwDNiFsYaB)<br />
* ResNet101 - [Google Drive](https://drive.google.com/open?id=1yMV6UmJZjbs1nshcXsyIQ1w0DU4M4apD)<br />
2) Adaptation from pascal voc to watercolor:<br />
* Resnet101 - [Google Drive](https://drive.google.com/open?id=1aH83DuNoT3YDCnPzW8OssDcKCw3SajAY)<br />
3) Adaptation from pascal voc to clipart:<br />
* Resnet101 - [Google Drive](https://drive.google.com/open?id=1d7jFLUnzhuj3xkCrAdiXXiay6QcGv3RZ)

### Train

We have provided sample training commands in train_scripts folder. However they are only for implementing our model.<br />
I am providing commands for implementing all three models below.
For SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses -:
```
CUDA_VISIBLE_DEVICES=$1 python trainval_net_SCL.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --save_dir $2
```
For Domain Adaptive Faster R-CNN for Object Detection in the Wild -: <br />
```
CUDA_VISIBLE_DEVICES=$1 python trainval_net_dfrcnn.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --save_dir $2
```
For Strong-Weak Distribution Alignment for Adaptive Object Detection -: <br />
```
CUDA_VISIBLE_DEVICES=$1 python trainval_net_global_local.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --gc --lc --save_dir $2
```

### Test

We have provided sample testing commands in test_scripts folder for our model. For others please take a reference of above training scripts. 

### Citation
If you use our code or find this helps your research, please cite:

```
@article{shen2019SCL,
  title={SCL: Towards Accurate Domain Adaptive Object Detection via
Gradient Detach Based Stacked Complementary Losses},
  author={Zhiqiang Shen and Harsh Maheshwari and Weichen Yao and Marios Savvides},
  journal={arXiv preprint arXiv:1911.02559},
  year={2019}
}
```

### Examples
<div align=center>
<img src="https://user-images.githubusercontent.com/3794909/65907453-66be7900-e392-11e9-996e-daa0d41ee78b.png" width="780">
</div>
 <div align=center>
Figure 1: Detection Results from Pascal VOC to Clipart.
</div> 
 
<div align=center>
<img src="https://user-images.githubusercontent.com/3794909/65907605-a71df700-e392-11e9-9f95-18d65ff4ceb7.png" width="780">
</div>
<div align=center>
Figure 2: Detection Results from Pascal VOC to Watercolor.
</div> 
