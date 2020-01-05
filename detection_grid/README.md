## GridMask Data Augmentation for COCO Object Detection

### Installation

Our implementation is based on maskrcnn-benchmark, the installation is the same as original maskrcnn-benchmark.
Please refer to [maskrcnn-benckmark](https://github.com/facebookresearch/maskrcnn-benchmark/) for detail installation.

### Requirements
We train this code under the following enviromnet:
```
Nvidia: 396.82, Cuda: V9.0.176
python: 3.6.5, pytorch: 1.0.0, torchvision: 0.2.0
GPU: 8xP100
```

### Training

To train FasterRCNN-R50-FPN, run the following commonds:
```
NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_net.py \
	--config-file configs/grid_faster_rcnn_R_50_FPN_4x.yaml
```
