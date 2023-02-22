# LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images 

## This is a PyTorch implementation of [LO-Det](https://ieeexplore.ieee.org/document/9390310/), YOLOv3+MobileNetv2, and YOLOv4.  

如果觉得万一有点用，欢迎引用我们的论文。谢谢。 

@ARTICLE{9390310,
  author={Z. {Huang} and W. {Li} and X. -G. {Xia} and H. {Wang} and F. {Jie} and R. {Tao}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-15},  
  doi={10.1109/TGRS.2021.3067470}}  
  
##10.9 更新预告
国庆弄了个知识蒸馏和量化压缩的版本，DOTA1.0_OBB目前mAP能到71.38，和我5月那个GGHL的工作持平了，丢了个会议，等审稿完代码会和v2版本一起放出来。
另外根据大家都建议修改了一些bug和做了一些优化，有问题的可以看issues或者邮件联系我。谢谢。

##一些解释和惯例的碎碎念  
代码这几天会分批陆续上传，因为有一些函数我重新命名和封装了（因为代码是基于更复杂的检测框架的，要从其中把LO-Det的部分单独拆出来）。模型权重字典的key我也需要重新转换一下。 
目前是一个粗糙版本，因为想着尽快开源让大家用，还没来得及重构和优化，可能会有一些bug，遇到任何问题都可以在issues里面留言或者给我发邮件（邮件地址论文里有，一作那个就是），我会尽最大努力帮你们跑通实验和复现结果。 
这个代码和我的另一个工作[NPMMR-Det](https://github.com/Shank2358/NPMMR-Det)用的是一个框架，大部分基础函数和用法都是一样的，有不明白的也可以看那个仓库。后续（有时间的话）会在那边仓库出教程和每行代码的注释。  
关于在嵌入式设备上实验以及Nvidia TX2\Xavier的其他使用问题也欢迎交流。  
最近事儿挺多，代码优化请允许我延迟一些时间（大概一个月之内），我会尽快完成。 
有计划出一版LO-Detv2的魔改版，正在做实验。这个做完以后接下来的目标检测工作可能会逐渐淡化遥感方向了，对各方面都有点失望吧。

## The High-precision version [NPMMR-Det](https://github.com/Shank2358/NPMMR-Det)

A Novel Nonlocal-aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images  

@ARTICLE{9364888,  
  author={Z. {Huang} and W. {Li} and X. -G. {Xia} and X. {Wu} and Z. {Cai} and R. {Tao}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={A Novel Nonlocal-Aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-20},  
  doi={10.1109/TGRS.2021.3059450}} 
  
Clone不Star，都是耍流氓~

## Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10, VS2019)   
CUDA 11.1, Cudnn 8.0.4

1. For RTX20/Titan RTX/V100 GPUs （I have tested it on RTX2080Ti, Titan RTX, and Tesla V100 (16GB)）  
cudatoolkit==10.0.130  
numpy==1.17.3  
opencv-python==3.4.2  
pytorch==1.2.0  
torchvision==0.4.0  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...  
The installation of other libraries can be carried out according to the prompts of pip/conda  
  
2. For RTX30 GPUs （I have tested it on RTX3080 and RTX3090 GPUs）  
cudatoolkit==11.0.221  
numpy==1.17.5  
opencv-python==4.4.0.46  
pytorch==1.7.0  
torchvision==0.8.1  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...

## Installation
1. git clone this repository    
2. Install the libraries in the ./lib folder  
(1) DCNv2  
cd ./NPMMR-Det/lib/DCNv2/  
sh make.sh  
(2) pycocotools  
cd ./NPMMR-Det/lib/cocoapi/PythonAPI/  
sh make.sh  

## Datasets
1. [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html) and its [devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
2. [DIOR dataset](https://pan.baidu.com/share/init?surl=w8iq2WvgXORb3ZEGtmRGOw), password: 554e  
(1) VOC Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./data folder.  
For the specific format of the train.txt file, see the example in the /data folder.  
(2) MSCOCO Format  
put the .json file in the ./data folder.

## Usage Example
1. train  
python train.py  
2. test  
python test.py  

## Parameter Settings
Modify ./cfg/cfg_npmmr.py, please refer to the comments in this file for details

## Weights
（https://drive.google.com/drive/folders/1cj0IKZ6rSVQRnt27AWiae0mSKhlLRQgs?usp=share_link）

## Notice
If you have any questions, please ask in issues.  
If you find bugs, please let me know and I will debug them in time. Thank you.  
I will do my best to help you run this program successfully and get results close to those reported in the paper.  

## Something New
In addition to YOLOv3, YOLOv4 has also been initially implemented in this repository.  
Some of the plug-and-play modules (many many Attentions, DGC, DynamicConv, PSPModule, SematicEmbbedBlock...) proposed in the latest papers are also collected in the ./model/plugandplay, you can use them and evaluate their performance freely. If it works well, please share your results here. Thank you.

## To Do
(1) Model Pruning  
(2) ONNX & TensorRT  
...  

## References
https://github.com/Shank2358/NPMMR-Det  
https://github.com/Peterisfar/YOLOV3  
https://github.com/argusswift/YOLOv4-pytorch  
https://github.com/ultralytics/yolov5  
https://github.com/pprp/SimpleCVReproduction  

## License
This project is released under the [Apache 2.0 license](LICENSE).
 
