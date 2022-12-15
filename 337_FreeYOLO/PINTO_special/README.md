# Note
- Demo - FreeYOLO Nano 192x320 PINTO_special - Corei9 Gen.10 CPU

  https://user-images.githubusercontent.com/33194443/207495316-31b4f2ff-7ba6-49ef-af79-a84c7d02e216.mp4

- Demo - FreeYOLO Nano 640x640 PINTO_special - Corei9 Gen.10 CPU

  https://user-images.githubusercontent.com/33194443/207490809-cd41b658-01dc-4bab-a02a-4092cc58f38b.mp4

- Post-Process (Myriad Support) - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/337_FreeYOLO/PINTO_special/convert_script.txt
![image](https://user-images.githubusercontent.com/33194443/207271656-0b7fc7ca-aadb-4d3c-b18c-388bd60c687d.png)

# How to change NMS parameter
## 0. Tools Install
### 0-1. Key Tools
```
pip install -U onnx \
&& pip install -U nvidia-pyindex \
&& pip install -U onnx-graphsurgeon \
&& pip install -U onnxsim \
&& pip install -U simple_onnx_processing_tools \
&& pip install -U onnx2tf
```
### 0-2. [Optional] onnxruntime, onnxruntime-gpu
```
sudo pip unisntall -y onnxruntime onnxruntime-gpu

pip install -U onnxruntime

or

pip install -U onnxruntime-gpu
```
or

- onnxruntime-gpu build (Generation of onnxruntime-gpu installers that match the CUDA version and TensorRT version of your environment)
  - If the build fails, OutOfMemory is most likely occurring. Therefore, change `--parallel $(nproc)` to a number such as `--parallel 4` or `--parallel 2` to adjust the number of parallel builds.
  - I dare to use TensorRT 8.4.0 EA because TensorRT 8.4.1+ has a problem that significantly degrades FP16 accuracy.
  - The version of onnxruntime obtained from GitHub is usually fine as long as it is a newer version.
  - For Windows:
  
    ```
    git clone -b v1.13.1 https://github.com/microsoft/onnxruntime.git
    cd onnxruntime

    pip install -U cmake

    .\build.bat ^
    --config=Release ^
    --cmake_generator="Visual Studio 16 2019" ^
    --build_shared_lib ^
    --cudnn_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4" ^
    --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4" ^
    --use_tensorrt ^
    --use_cuda ^
    --cuda_version 11.4 ^
    --tensorrt_home "C:\Program Files\TensorRT-8.4.0.6" ^
    --enable_pybind ^
    --build_wheel ^
    --enable_reduced_operator_type_support ^
    --skip_tests
    
    pip uninstall onnxruntime onnxruntime-gpu
    pip install -U .\build\Windows\Release\Release\dist\onnxruntime_gpu-1.13.1-cp39-cp39-win_amd64.whl
    ```
  - For Linux:

    ```
    git clone https://github.com/microsoft/onnxruntime.git \
    && cd onnxruntime
    git checkout 49d7050b88338dd57839159aa4ce8fb0c199b064

    dpkg -l | grep TensorRT

    ii graphsurgeon-tf        8.4.0-1+cuda11.6   amd64 GraphSurgeon for TensorRT package
    ii libnvinfer-bin         8.4.0-1+cuda11.6   amd64 TensorRT binaries
    ii libnvinfer-dev         8.4.0-1+cuda11.6   amd64 TensorRT development libraries and headers
    ii libnvinfer-doc         8.4.0-1+cuda11.6   all   TensorRT documentation
    ii libnvinfer-plugin-dev  8.4.0-1+cuda11.6   amd64 TensorRT plugin libraries
    ii libnvinfer-plugin8     8.4.0-1+cuda11.6   amd64 TensorRT plugin libraries
    ii libnvinfer-samples     8.4.0-1+cuda11.6   all   TensorRT samples
    ii libnvinfer8            8.4.0-1+cuda11.6   amd64 TensorRT runtime libraries
    ii libnvonnxparsers-dev   8.4.0-1+cuda11.6   amd64 TensorRT ONNX libraries
    ii libnvonnxparsers8      8.4.0-1+cuda11.6   amd64 TensorRT ONNX libraries
    ii libnvparsers-dev       8.4.0-1+cuda11.6   amd64 TensorRT parsers libraries
    ii libnvparsers8          8.4.0-1+cuda11.6   amd64 TensorRT parsers libraries
    ii onnx-graphsurgeon      8.4.0-1+cuda11.6   amd64 ONNX GraphSurgeon for TensorRT package
    ii python3-libnvinfer     8.4.0-1+cuda11.6   amd64 Python 3 bindings for TensorRT
    ii python3-libnvinfer-dev 8.4.0-1+cuda11.6   amd64 Python 3 development package for TensorRT
    ii tensorrt               8.4.0.6-1+cuda11.6 amd64 Meta package of TensorRT
    ii uff-converter-tf       8.4.0-1+cuda11.6   amd64 UFF converter for TensorRT package

    sudo chmod +x build.sh
    sudo pip install cmake --upgrade

    ./build.sh \
    --config Release \
    --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --cuda_home /usr/local/cuda \
    --use_tensorrt \
    --use_cuda \
    --tensorrt_home /usr/src/tensorrt/ \
    --enable_pybind \
    --build_wheel \
    --parallel $(nproc) \
    --skip_tests

    find . -name "*.whl"
    ./build/Linux/Release/dist/onnxruntime_gpu-1.12.0-cp38-cp38-linux_x86_64.whl

    sudo pip uninstall onnxruntime onnxruntime-gpu
    pip install ./build/Linux/Release/dist/onnxruntime_gpu-*.whl
    ```

## 1. max_output_boxes_per_class
  Value that controls the maximum number of bounding boxes output per class. The closer the value is to 1, the faster the overall post-processing speed.
  ```
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
  or
  ```
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
## 2. iou_threshold
  NMS IOU Thresholds.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants iou_threshold float32 [0.5]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants iou_threshold float32 [0.5]
  ```
## 3. score_threshold
  Threshold of scores to be detected for banding boxes. The closer the value is to 1.0, the faster the overall post-processing speed.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants score_threshold float32 [0.75]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants score_threshold float32 [0.75]
  ```
