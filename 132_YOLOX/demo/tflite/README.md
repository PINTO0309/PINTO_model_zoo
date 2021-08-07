# YOLOX with TensorFlow Lite - Python

## Description
This sample contains Python code that running YOLOX-TensorFlow Lite model.

## Reference
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  - [YOLOX-ONNXRuntime in Python](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime)  
- [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
  - [YOLOX models](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/132_YOLOX)

## Environment
- HW
  - PC or Raspberry Pi
  - Camera (optional)
- SW
  - TensorFlow Lite Python Interpreter v2.5
  - OpenCV Python v4.x (maybe v3.x)

## How to

Clone this repository.
```
cd ~
git clone https://github.com/PINTO0309/PINTO_model_zoo.git
cd ./PINTO_model_zoo
git submodule init && git submodule update
```

Download YOLOX models.
```
cd ./132_YOLOX/

# YOLOX Nano
download_nano.sh

# YOLOX Tiny
download_tiny.sh
```
Run demo.
```
cd ~/tflite-cv-exampl/eyolox/python

# Camera input demo.
python yolox_tflite_demo.py \
  --model PATH_TO_TFLITE_MODEL_FILE \
  --label ./data/coco_labels.txt

# Video file input demo.
python yolox_tflite_demo.py 
  --model PATH_TO_TFLITE_MODEL_FILE \
  --label ../data/coco_labels.txt \
  --videopath PATH_TO_VIDEO_FILE
```

## LICENSE

The following files are licensed under [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

- 132_YOLOX/demo/tflite/yolox/utils/demo_utils.py