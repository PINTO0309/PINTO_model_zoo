# OpenVINO msg_chn_wacv20 depth completion
Python script for performing depth completion from sparse depth and rgb images using the msg_chn_wacv20 model in OpenVINO. The example takes a synthetic depth map, it reduces the density (variable) of the depthmap and passes it to the depth completion map to densify the depth map.

# Requirements

 * **OpenCV**, **OpenVINO**. Also, **unrealcv** is only required if you want to generate new data using unrealcv.

# UnrealCV synthethic data generation
The input images and depth are generated using the UnrealCV library (https://unrealcv.org/), you can find more information about how to generate this data in this [other repository for Unreal Synthetic depth generation](https://github.com/ibaiGorordo/UnrealCV-stereo-depth-generation).

# Installation
```bash
pip install -r requirements.txt
```

# Download OpenVINO models and test data
```bash
# OpenVINO models
$ gdown \
--id 1f0_FHJ2vmDWAPhN4pPu2gvl_Mk_ufDx9 \
--output models/resources.tar.gz
$ tar -zxvf models/resources.tar.gz -C models && rm models/resources.tar.gz

# Test data
$ gdown \
--id 1M8ZgA11ZCeJhdIgtlrMLzs9GVxFDYweQ \
--output outdoor_example/resources.tar.gz
$ tar -zxvf outdoor_example/resources.tar.gz -C outdoor_example && rm outdoor_example/resources.tar.gz
```

# Examples

  * **Video inference (UnrealCV synthetic data)**:

 ```bash
 python video_depth_estimation_openvino.py
 ```

# Special Thanks (Citation)
- https://github.com/ibaiGorordo/ONNX-msg_chn_wacv20-depth-completion
- https://github.com/ibaiGorordo/TFLite-msg_chn_wacv20-depth-completion