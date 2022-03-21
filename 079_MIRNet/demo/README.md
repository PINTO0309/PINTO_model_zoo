# Demo projects

## MIRNet with TensorRT in Python
https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/mirnet/README.md

## MIRNet with ONNX in Python
```bash
##### For GPU
xhost +local: && \
docker run --gpus all -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ghcr.io/pinto0309/openvino2tensorflow:latest

##### For CPU
xhost +local: && \
docker run -it --rm \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ghcr.io/pinto0309/openvino2tensorflow:latest

##### Run
python demo_MIRNet_onnx.py
```
If you want to change the model, specify it with an argument.
```python
    parser.add_argument(
        "--device",
        type=int,
        default=0
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_360x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='360,640',
    )
```
https://user-images.githubusercontent.com/33194443/159192316-0e6eb6b7-0e5c-4c7f-8daa-b7453d5becde.mp4

