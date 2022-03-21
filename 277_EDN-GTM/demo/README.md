# Demo projects

## EDN-GTM with ONNX Runtime in Python
```
##### Environment
https://github.com/PINTO0309/openvino2tensorflow#1-environment

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
python demo_EDN-GTM_onnx.py
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
        default='ihaze_generator_384x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='384,640',
    )
```

- ONNX 384x640 - Input movie 360x640 - ONNX TensorRT Execution Provider Float16  

  https://user-images.githubusercontent.com/33194443/159295239-af8e6f1f-c619-4847-96f8-da546406e347.mp4
