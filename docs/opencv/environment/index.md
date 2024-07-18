# 环境配置

## 一、安装 Python

[Python 安装地址](https://www.python.org/downloads/)

![alt text](2277a377bef441cfab59ee83aed3d148.png)
![alt text](3a387bbddf5f4128a6c5b1d62169ab7b.png)

???+ "提示"
    注意 Add PATH

![alt text](d441666b32954a4ba769e5433ac83831.png)

完成。

## 二、Pycharm

[pycharm windows 版本下载地址](https://www.jetbrains.com/pycharm/download/#section=windows)

选择社区版下载安装（日常学习使用够用了）（pycharm community edition）

![alt text](6710f0fb02d246479873c6d7bbf4ab39.png)
![alt text](6710f0fb02d246479873c6d7bbf4ab39-1.png)
![alt text](1f22813ba0614dddafef6554a6edf463.png)

可自选路径

![alt text](5796a425ced840fca236c86e7346045a.png)
![alt text](cbdb810ad6764f9a8f753e3947ee9da5.png)

√这三个即可

![alt text](316e8aa079814f1184d2b84229c30907.png)

重启

![alt text](96dbf898c5e1497985c543adbd933549.png)

如果之前安装过又卸载的出现这个，选择 do not 就好。

![alt text](ea0d25e2db2742e6b0f1c3ad47da44db.png)

New project

![alt text](b7abaa7ed2c04cc0ac3e6da0a2a889da.png)

可自行选择路径

![alt text](eb102eb40c9a4ea798def3e1f29b90a2.png)

可以简单测试一下是否能运行

设置解释器：

![alt text](970bc9aee28e485d993b402c40e460ae.png)
![alt text](5c244641dd2d4bb9ad660074c908dae5.png)

目前是使用的虚拟解释器，因为我们不涉及管理不同版本的 Python，所以我们将其设置成我们刚安装的系统解释器，最为简单方便

点击右上角的 Add

![alt text](5babcabaf1a1491ab749911639bdf052.png)

选择 System interpreter

![alt text](90ebac3e7d4746ffbfd282ffd7a305d6.png)
![alt text](6ce2a82f174b46c9a5e76e78d9fb1ba8.png)

Apply and OK 即可

中文包：

![alt text](56596919c93244ab8f7222021699ccc4.png)

## 三、Anaconda 环境

### 1、conda 的安装

[conda 安装（清华大学开源镜像站）](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

![alt text](0abbab05c3094d88a2d023c040cf112f.png)

根据自己电脑选择合适版本

![alt text](98313cac5c8841b9813107844bc36680.png)
![alt text](143105640bbd4f4e97034bb562612696.png)
![alt text](ccf6752ef27e4954b8e9436d8f3cbf53.png)

这样我们的anaconda就安装好了。

![alt text](02054d49d99044418cf72bcff7bc34c2.png)

打开pycharm，使用conda环境新建项目

![alt text](74151e8d24ba482fb1000d5f282e1161.png)

在设置中查看解释器：

![alt text](720bd936ae3146518c3bbe3c4c627343.png)

添加解释器可添加自己创建的虚拟环境：

![alt text](0138556b39974af9b7c5df7fde014155.png)

### 2、创建 conda 虚拟环境的方法

打开 Anaconda Prompt

![alt text](106638d424ce455287c6083b777186c7.png)

```cmd
conda create -n xxx python=3.8
# xxx改成你的虚拟环境名字 python=版本号
conda activate xxx
# 激活创建的虚拟环境
```

![alt text](80d1fbb2491248a9bd7e3ba2d3f0e407.png)

前面的 base 变成你创建的虚拟环境名称即为成功。

[具体conda使用的常用指令可以参考大神的博客](http://t.csdnimg.cn/KNZfl)

## 四、OpenCV

先安装 numpy

`pip install numpy`

pip 安装

`pip install opencv-python`
`pip install opencv-contrib-python`

验证一下有没有安装成功

```cmd
C:\Users\kevin>python
Python 3.11.7 | packaged by Anaconda, Inc. | (main, Dec 15 2023, 18:05:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> print(cv2.__version__)
4.9.0
```

## 五、Pytorch（深度学习环境）

前提：你的电脑有GPU

### 1、cmd 终端输入 nvidia-smi，查看你的显卡适配的 CUDA 版本

![alt text](152f60a18af34357b8035b1f77ee68cd.png)

### 2、conda 创建虚拟环境

Anaconda Prompt 输入

`conda create -n pytorch python=3.8`

`conda activate pytorch`

### 3、下载地址： [PyTorch](https://pytorch.org/)

![alt text](c781da6647b440a5a76a7618c7161038.png)

根据你电脑支持的 CUDA 版本安装，我的电脑是 CUDA12.1

`(pytorch) C:\Users\kevin>conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

验证一下：

```cmd
python
import torch
torch.cuda.is_available()
```

![alt text](e0729b6a9b5a448f8e977b555c6ff389.png)

如果返回值是 True，代表成功啦。恭喜你，安装完成，可以开始快乐的炼丹啦。

## 六、yolov8

[ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > OpenVINO > CoreML > TFLite (github.com)](https://github.com/ultralytics/ultralytics)

根据官方的 README 安装使用

Install

>Pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

在有 pytorch 的 conda 虚拟环境中输入

`pip install ultralytics`

跑一个官方例程测试

`yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'`

恭喜你，完成了 yolov8 的安装。
