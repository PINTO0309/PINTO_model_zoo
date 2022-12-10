# Demo projects

## DAMO-YOLO with ONNX Runtime in Python
```
python demo_DAMO-YOLO_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='damoyolo_tinynasL20_T_192x320.onnx',
    )
```
<br>
A model with post-processing runs like this.<br>

```
python demo_DAMO-YOLO_with_postprocess_onnx.py
```

If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--model",
        type=str,
        default='damoyolo_tinynasL20_T_192x320_post.onnx',
    )
```

- TensorRT + `demo_DAMO-YOLO_with_postprocess_onnx.py` + `damoyolo_tinynasL35_M_384x640_post.onnx`

    https://user-images.githubusercontent.com/33194443/206323392-28ac9707-8c12-4431-870b-a90399955b83.mp4

- onnxruntime-gpu build (Generation of onnxruntime-gpu installers that match the CUDA version and TensorRT version of your environment)
  - If the build fails, OutOfMemory is most likely occurring. Therefore, change `--parallel $(nproc)` to a number such as `--parallel 4` or `--parallel 2` to adjust the number of parallel builds.
  - I dare to use TensorRT 8.4.0 EA because TensorRT 8.4.1+ has a problem that significantly degrades FP16 accuracy.

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
