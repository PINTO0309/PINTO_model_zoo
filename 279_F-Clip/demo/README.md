# Demo projects

## F-Clip with ONNX Runtime in Python
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
python demo_F-Clip_onnx.py
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
        default='fclip_hr_512x512/fclip_hr_512x512.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,512',
    )
    parser.add_argument(
        "--provider",
        choices=[
            'gpu',
            'openvino',
            'cpu'
        ],
        default='gpu',
    )
```

- F-Clip HR, ONNX 512x512 - USB Camera 480x640 - ONNX TensorRT Execution Provider Float16

https://user-images.githubusercontent.com/33194443/159731723-a2840ca1-a6b5-4175-90dc-41a8238927e9.mp4
