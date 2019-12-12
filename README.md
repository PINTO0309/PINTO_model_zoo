# PINTO_model_zoo

## 1. Environment

- Ubuntu 18.04
- Tensorflow-GPU v1.15.0
- Python 3.6.8

## 2. Procedure
```bash
$ cd ~
$ mkdir deeplab;cd deeplab
$ git clone --depth 1 https://github.com/tensorflow/models.git
$ cd models/research/deeplab/datasets
$ mkdir pascal_voc_seg

$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rATNHizJdVHnaJtt-hW9MOgjxoaajzdh" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rATNHizJdVHnaJtt-hW9MOgjxoaajzdh" -o pascal_voc_seg/VOCtrainval_11-May-2012.tar

$ sed -i -e "s/python .\/remove_gt_colormap.py/python3 .\/remove_gt_colormap.py/g" \
      -i -e "s/python .\/build_voc2012_data.py/python3 .\/build_voc2012_data.py/g" \
      download_and_convert_voc2012.sh

$ sh download_and_convert_voc2012.sh

$ cd ../..
$ mkdir -p deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/train
$ mkdir -p deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/eval
$ mkdir -p deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/vis

$ export PATH_TO_TRAIN_DIR=${HOME}/deeplab/models/research/deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/train
$ export PATH_TO_DATASET=${HOME}/deeplab/models/research/deeplab/datasets/pascal_voc_seg/tfrecord
$ export PYTHONPATH=${HOME}/deeplab/models/research:${HOME}/deeplab/models/research/deeplab:${HOME}/deeplab/models/research/slim:${PYTHONPATH}
```

```python
# See feature_extractor.network_map for supported model variants.
# models/research/deeplab/core/feature_extractor.py

networks_map = {
    'mobilenet_v2': _mobilenet_v2,
    'mobilenet_v3_large_seg': mobilenet_v3_large_seg,
    'mobilenet_v3_small_seg': mobilenet_v3_small_seg,
    'resnet_v1_18': resnet_v1_beta.resnet_v1_18,
    'resnet_v1_18_beta': resnet_v1_beta.resnet_v1_18_beta,
    'resnet_v1_50': resnet_v1_beta.resnet_v1_50,
    'resnet_v1_50_beta': resnet_v1_beta.resnet_v1_50_beta,
    'resnet_v1_101': resnet_v1_beta.resnet_v1_101,
    'resnet_v1_101_beta': resnet_v1_beta.resnet_v1_101_beta,
    'xception_41': xception.xception_41,
    'xception_65': xception.xception_65,
    'xception_71': xception.xception_71,
    'nas_pnasnet': nas_network.pnasnet,
    'nas_hnasnet': nas_network.hnasnet,
}
```
### 2-1. "mobilenet_v3_small_seg" Float32 regular training
```bash
$ python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=500000 \
    --train_split="train" \
    --model_variant="mobilenet_v3_small_seg" \
    --decoder_output_stride=16 \
    --train_crop_size="513,513" \
    --train_batch_size=8 \
    --dataset="pascal_voc_seg" \
    --save_interval_secs=300 \
    --save_summaries_secs=300 \
    --save_summaries_images=True \
    --log_steps=100 \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

## 3. Visualize training status
```bash
$ tensorboard --logdir ${HOME}/deeplab/models/research/deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/train
```
