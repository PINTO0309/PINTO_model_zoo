# Note

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b349cc3c-2586-4070-9c7b-a8bbce0b22b2

```
usage:
demo_opal23_headpose_onnx.py \
[-h] \
[-dm DETECTION_MODEL] \
[-hm HEADPOSE_MODEL] \
(-v VIDEO | -i IMAGES_DIR) \
[-ep {cpu,cuda,tensorrt}] \
[-dvw] \
[-dwk]

options:
  -h, --help
    show this help message and exit
  -dm DETECTION_MODEL, --detection_model DETECTION_MODEL
    ONNX/TFLite file path for ObjectDetection.
  -hm HEADPOSE_MODEL, --headpose_model HEADPOSE_MODEL
    ONNX/TFLite file path for HeadPose.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -i IMAGES_DIR, --images_dir IMAGES_DIR
    jpg, png images folder path.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic recording to MP4.
    Devices that use a MicroSD card or similar for main storage can speed up overall processing.
  -dwk, --disable_waitKey
    Disable cv2.waitKey(). When you want to process a batch of still images, disable key-input
    wait and process them continuously.
```
