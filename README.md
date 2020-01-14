# GridMask Data Augmentation

## Introduction
This project is the code for implementing the GridMask augmentation for image classification and object detection.
The full paper is available at: [https://arxiv.org/abs/2001.04086](http://arxiv.org/abs/2001.04086). 

## Main Result

ImageNet

|                | Baseline   | + GridMask   |
|----------------|------------|--------------|
| ResNet-50      | 76.5       | 77.9         |
| ResNet-101     | 78.0       | 79.1         |
| ResNet-152     | 78.3       | 79.7         |

COCO2017

|                      | Baseline   | + GridMask   |
|----------------------|------------|--------------|
| FasterRCNN-R50-FPN   | 37.4       | 39.2         |
| FasterRCNN-X101-FPN  | 41.2       | 42.6         |

You can find pretrained models to achieve the results in [models](https://drive.google.com/drive/folders/12Vs8i3OrafXV5NjzuJVCHsa5i4oF9gmu?usp=sharing)

