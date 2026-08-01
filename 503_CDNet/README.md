# 503_CDNet

### Paper
[Combined Depth Space based Architecture Search for Person Re-identification](https://arxiv.org/abs/2104.04163)

<img width="853" height="571" alt="image" src="https://github.com/user-attachments/assets/1e3f3171-1c70-4bc5-bad2-64e2d39a0243" />

### Models

- results on ReID tasks

  | model             | Market(mAP/rank-1) | Duke(mAP/rank-1) | MSMT17(mAP/rank-1) |
  | ----------------- | :----------------: | :--------------: | :----------------: |
  | cnet(scratch)     |     83.5/93.6      |    73.2/86.0     |     47.7/73.3      |
  | cdnet(scratch)    |     83.7/93.7      |    73.9/86.7     |     48.5/73.7      |
  | cdnet(pretrained) |     86.0/95.1      |    76.8/88.6     |     54.7/78.9      |

  - results on classification

  | model          | Cifar-100(acc/param) | ImageNet(acc/param) |
  | -------------- | -------------------- | ------------------- |
  | cdnet(scratch) | 82.1/2.3M            | 75.1/2.5M           |

## Original

https://github.com/solicucu/ReID

## ONNX custom

https://github.com/PINTO0309/ReID
