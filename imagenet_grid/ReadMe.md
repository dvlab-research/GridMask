## GridMask Data Augmentation for ImageNet Classification

### Installation

We use the offical ImageNet training code in Apex. 
Please refer to [Apex](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) for detail installation.

### Requirements
We train this code under the following enviromnet:
```
Nvidia: 418.67, Cuda: V10.0.130
python: 3.6.5, pytorch: 1.3.0+cu100, torchvision: 0.4.1+cu100
GPU: 8xP40
```
Other pytorch vision should also work.

### Training

To train ResNet-50, run the following commonds:
```
python -m torch.distributed.launch --nproc_per_node=8 imagenet_amp.py -a resnet50 \
    --opt-level O0 --b 32 -p 1000 -j 64 --grid \
    --epochs 300 --lr_adjust_step 100 200 265 \
    --save_dir checkpoints/resnet50 \
    path-to-ImageNet
```
