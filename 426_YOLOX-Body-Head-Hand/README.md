# YOLOX-Body-Head-Hand

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection model generated using a high-quality human dataset. I annotated all the data by myself. Extreme resistance to blur and occlusion. In addition, the recognition rate at short, medium, and long distances has been greatly enhanced. The camera's resistance to darkness and halation has been greatly improved.

`Head` does not mean `Face`. Thus, the entire head is detected rather than a narrow region of the face. This makes it possible to detect all 360° head orientations.

  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/423_6DRepNet360

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/ab4c4b1b-6e51-416a-948f-809b3d06eafd

The advantage of being able to detect hands with high accuracy is that it makes it possible to detect key points on the fingers as correctly as possible. The video below is processed by converting the MediaPipe tflite file to ONNX, so the performance of keypoint detection is not very high. It is assumed that information can be acquired quite robustly when combined with a highly accurate keypoint detection model focused on the hand region. It would be realistic to use the distance in the Z direction, which represents depth, in combination with physical information such as ToF, rather than relying on model estimation. To obtain as accurate a three-dimensional value as possible, including depth, sparse positional information on a two-dimensional plane, such as skeletal detection, is likely to break down the algorithm. This has the advantage that unstable depths can be easily corrected by a simple algorithm by capturing each part of the body in planes, as a countermeasure to the phenomenon that when information acquired from a depth camera (ToF or stereo camera parallax measurement) is used at any one point, the values are affected by noise and become unstable due to environmental noise.

The method of detecting 133 skeletal keypoints at once gives the impression that the process is very heavy because it requires batch or loop processing to calculate heat maps for multiple human bounding boxes detected by the object detection model. I also feel that the computational cost is high because complex affine transformations and other coordinate transformation processes must be performed on large areas of the entire body. However, this is not my negative view of a model that detects 133 keypoints, only that it is computationally expensive to run on an unpowered edge device.

  https://github.com/PINTO0309/hand_landmark

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b240f9af-4f17-4b02-ba62-2b12838510ce

## 1. Dataset
  - COCO-Hand (14,667 Images, 66,903 labels, All re-annotated manually)
  - http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip
  - I am adding my own enhancement data to COCO-Hand and re-annotating all images. In other words, only COCO images were cited and no annotation data were cited.
  - I have no plans to publish my own dataset.
    ```
    body_label_count: 30,729 labels
    head_label_count: 26,268 labels
    hand_label_count: 18,087 labels
    ===============================
               Total: 66,903 labels
               Total: 14,667 images
    ```
    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/22b56779-928b-44d8-944c-25431b83e24f)
  - The process of work and verification can be seen in my working notes below. However, a detailed explanation is not given.

    https://zenn.dev/pinto0309/scraps/11300d816ab1b3

## 2. Annotation

  Halfway compromises are never acceptable.

  ![000000000544](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/557b932b-767b-4f8c-87f5-75f403fa9c50)

  ![000000000716](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9acb2308-eba1-4a05-91ed-ccbb6e122f67)

  ![000000002470](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c1809eeb-7b2c-41de-a519-9834c804c656)

  ![icon_design drawio (3)](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/72740ed3-ae9f-4ab7-9b20-bea62c58c7ac)

## 3. Test
  - Python 3.10
  - onnx 1.14.1+
  - onnxruntime-gpu v1.16.1 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.16.1 + CUDA 11.8 + TensorRT 8.5.3 build (RTX3070)](https://zenn.dev/pinto0309/scraps/20afd3c58b30bf))
  - opencv-contrib-python 4.8.0.76
  - numpy 1.24.3
  - TensorRT 8.5.3-1+cuda11.8
  - [tflite-runtime](https://github.com/PINTO0309/TensorflowLite-bin) v2.15.0+
  - TensorFlow v2.15.0+

    ```bash
    # Common ############################################
    pip install opencv-contrib-python numpy onnx
  
    # For ONNX ##########################################
    pip uninstall onnxruntime onnxruntime-gpu
  
    pip install onnxruntime
    or
    pip install onnxruntime-gpu
  
    # For ARM. tflite_runtime ###########################
    TFVER=2.15.0.post1
  
    PYVER=310
    or
    PYVER=311
  
    ARCH=aarch64
    or
    ARCH=armhf
  
    pip install \
    --no-cache-dir \
    https://github.com/PINTO0309/TensorflowLite-bin/releases/download/v${TFVER}/tflite_runtime-${TFVER/-/}-cp${PYVER}-none-linux_${ARCH}.whl
  
    # For x86/x64. TensorFlow ############################
    pip install tensorflow
    ```

  - Demonstration of models with built-in post-processing (Float32/Float16)
    ```
    usage:
      demo_yolox_onnx_tfite.py \
      [-h] \
      [-m MODEL] \
      [-v VIDEO] \
      [-ep {cpu,cuda,tensorrt}] \
      [-dvw]
  
    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for YOLOX.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -dvw, --disable_video_writer
        Eliminates the file I/O load associated with automatic recording to MP4.
        Devices that use a MicroSD card or similar for main storage can speed up overall processing.
    ```
  - Demonstration of a model without built-in post-processing (UINT8/INT8/Float32/Float16)
    ```
    usage:
      demo_yolox-ti_onnx_tflite.py \
      [-h] \
      [-m MODEL] \
      [-st SCORE_THRESHOLD] \
      [-it IOU_THRESHOLD] \
      [-v VIDEO] \
      [-ep {cpu,cuda}] \
      [-dvw]
    
    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for YOLOX.
      -st SCORE_THRESHOLD, --score_threshold SCORE_THRESHOLD
        NMS score threshold.
      -it IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
        NMS IoU threshold.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -ep {cpu,cuda}, --execution_provider {cpu,cuda}
        Execution provider for ONNXRuntime.
      -dvw, --disable_video_writer
        Disable video writer. Eliminates the file I/O load associated with automatic recording to MP4.
        Devices that use a MicroSD card or similar for main storage can speed up overall processing.
    ```

  - 640x480 TensorRT

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx \
    -v 0 \
    -ep tensorrt
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/7017bc13-4bf4-4df4-855e-67af3344e3c3

  - 320x256 CPU Corei9, UINT8 quantized ONNX + onnxruntime

    ```bash
    python demo/demo_yolox-ti_onnx_tflite.py \
    -m yolox_ti_body_head_hand_n_1x3x256x320_uint8.onnx \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/a7239ab9-2410-40bb-b5e6-6ee2263a905e

  - 320x256 CPU Corei9, Float32 ONNX + onnxruntime

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/5d14b76e-daea-473f-8730-e3b0da0d0164

  - 160x128 CPU Corei9, Float32 ONNX + onnxruntime

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_s_body_head_hand_post_0299_0.4983_1x3x128x160.onnx \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9be1e3e0-afba-45b7-8f4f-24f4fb7e9340

  - [Gold-YOLO-n 320x256 CPU](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/425_Gold-YOLO-Body-Head-Hand) vs YOLOX-Nano 320x256 CPU

    |NMS param|value|
    |:-|-:|
    |max_output_boxes_per_class|20|
    |iou_threshold|0.40|
    |score_threshold|0.25|

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9444e5ab-a8d3-4290-81e7-7d1701abca16

  - 640x384 CUDA

    |NMS param|value|
    |:-|-:|
    |max_output_boxes_per_class|100|
    |iou_threshold|0.50|
    |score_threshold|0.25|

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_m_body_head_hand_post_0299_0.5263_1x3x384x640.onnx \
    -v pexels_videos_2670_640x360.mp4 \
    -ep cuda
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8a15e0d6-5f75-4246-adf5-478cb45d74b6

  - 640x384 TensorRT

    |NMS param|value|
    |:-|-:|
    |max_output_boxes_per_class|100|
    |iou_threshold|0.50|
    |score_threshold|0.25|

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_m_body_head_hand_post_0299_0.5263_1x3x384x640.onnx \
    -v pexels_videos_2670_640x360.mp4 \
    -ep tensorrt
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c10ad469-a99e-4ea3-9b71-047a3d19d9f8

  - 320x256 TFLite XNNPACK CPU x86/x64

    |NMS param|value|
    |:-|-:|
    |max_output_boxes_per_class|20|
    |iou_threshold|0.40|
    |score_threshold|0.25|

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/f09aa66b-e6c9-4c4e-a886-e80748a48882

  - 160x128 Float32 TFLite XNNPACK CPU RaspberryPi4 Bookworm GUI mode

    |NMS param|value|
    |:-|-:|
    |max_output_boxes_per_class|20|
    |iou_threshold|0.40|
    |score_threshold|0.25|

    https://github.com/PINTO0309/TensorflowLite-bin

    ```bash
    python demo/demo_yolox_onnx_tfite.py \
    -m yolox_n_body_head_hand_post_0461_0.4428_1x3x128x160_float32.tflite \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/1013816e-4665-41b8-abd7-e479d32f7763

  - 160x128 INT8 TFLite CPU RaspberryPi4 Bookworm GUI mode, 4 threads

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/df34dd58-8181-42c8-b03b-5aca93ed35ee

  - 160x128 Float16 TFLite CPU RaspberryPi Zero 2 W Bullseye armhf GUI mode, 4 threads

    ```bash
    134.0ms/pred
    ```

  - 160x128 Float16 TFLite CPU RaspberryPi Zero 2 W Bullseye aarch64 GUI mode, 4 threads

    ```bash
    115.0ms/pred
    ```

  - 160x128 INT8 TFLite CPU RaspberryPi Zero 2 W Bullseye aarch64 GUI mode, 4 threads

    ```bash
    81ms/pred
    ```
    ```
    python demo/demo_yolox-ti_onnx_tflite.py \
    -m yolox_ti_body_head_hand_n_1x3x128x160_bgr_uint8.tflite \
    -v 0
    ```

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/ba690884-9b94-4356-8a43-82a1161ea5b1

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/7e01e98c-9d71-4c3b-9663-a0743af49b64)

- Body-Head-Hand - Nano
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.721
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.409
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.135
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.382
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.495
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.647
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.738
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 40.412 | head    | 49.183 | hand    | 33.564 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 48.985 | head    | 54.212 | hand    | 45.345 |
  ```

- Body-Head-Hand - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.762
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.473
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.318
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.606
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.723
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.143
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.405
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.674
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 45.962 | head    | 51.465 | hand    | 38.975 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 53.025 | head    | 56.050 | hand    | 48.280 |
  ```

- Body-Head-Hand - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.793
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.525
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.344
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.674
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.809
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.154
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.445
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.432
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.735
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.844
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 52.107 | head    | 54.694 | hand    | 43.127 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 58.811 | head    | 59.010 | hand    | 51.898 |
  ```

- Body-Head-Hand - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.528
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.812
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.558
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.705
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.863
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.160
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.466
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.586
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.760
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.890
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 56.232 | head    | 55.763 | hand    | 46.546 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 62.242 | head    | 59.877 | hand    | 53.642 |
  ```

- Body-Head-Hand - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.542
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.816
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.569
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.720
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.879
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.474
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.452
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.904
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 57.914 | head    | 56.578 | hand    | 48.080 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 63.232 | head    | 60.400 | hand    | 54.363 |
  ```

- Body-Head-Hand - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.829
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.586
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.389
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.734
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.886
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.164
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.484
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.463
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.778
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.912
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 60.640 | head    | 56.730 | hand    | 49.516 |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 65.332 | head    | 60.171 | hand    | 55.462 |
  ```

- Post-Process

  Because I add my own post-processing to the end of the model, which can be inferred by TensorRT, CUDA, and CPU, the benchmarked inference speed is the end-to-end processing speed including all pre-processing and post-processing. EfficientNMS in TensorRT is very slow and should be offloaded to the CPU.

  - NMS default parameter

    |param|value|note|
    |:-|-:|:-|
    |max_output_boxes_per_class|20|Maximum number of outputs per class of one type. `20` indicates that the maximum number of people detected is `20`, the maximum number of heads detected is `20`, and the maximum number of hands detected is `20`. The larger the number, the more people can be detected, but the inference speed slows down slightly due to the larger overhead of NMS processing by the CPU. In addition, as the number of elements in the final output tensor increases, the amount of information transferred between hardware increases, resulting in higher transfer costs on the hardware circuit. Therefore, it would be desirable to set the numerical size to the minimum necessary.|
    |iou_threshold|0.40|A value indicating the percentage of occlusion allowed for multiple bounding boxes of the same class. `0.40` is excluded from the detection results if, for example, two bounding boxes overlap in more than 41% of the area. The smaller the value, the more occlusion is tolerated, but over-detection may increase.|
    |score_threshold|0.25|Bounding box confidence threshold. Specify in the range of `0.00` to `1.00`. The larger the value, the stricter the filtering and the lower the NMS processing load, but in exchange, all but bounding boxes with high confidence values are excluded from detection. This is a parameter that has a very large percentage impact on NMS overhead.|

  - Change NMS parameters

    Use **[PINTO0309/sam4onnx](https://github.com/PINTO0309/sam4onnx)** to rewrite the `NonMaxSuppression` parameter in the ONNX file.

    For example,
    ```bash
    pip install onnxsim==0.4.33 \
    && pip install -U simple-onnx-processing-tools \
    && pip install -U onnx \
    && python -m pip install -U onnx_graphsurgeon \
        --index-url https://pypi.ngc.nvidia.com

    ### max_output_boxes_per_class
    ### Example of changing the maximum number of detections per class to 100.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --output_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --output_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --output_onnx_file_path yolox_s_body_head_hand_post_0299_0.4983_1x3x256x320.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/0135a005-8a79-4358-bd90-a468d44851ac)

- [TFLite] ARMv8.2 XNNPACK Float16 boost

  If you have an ARM CPU with an ARMv8.2 or higher version instruction set, you can use Float16 tflite committed in this repository to infer twice as fast as in the past. This is due to the significant contribution of XNNPACK.

  https://github.com/PINTO0309/onnx2tf/tree/main#cli-parameter

  https://github.com/PINTO0309/onnx2tf/pull/553

  https://blog.tensorflow.org/2023/11/half-precision-inference-doubles-on-device-inference-performance.html

  ```
  pip install -U onnx2tf
  onnx2tf -i yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320.onnx -eatfp16
  ```

  This is result in macbook.

  - Chip: Apple M1 Pro (ArmV8 processor, ARMv8.6A)
  - Python 3.9
  - Tensorflow 2.15.0 (from pip install tensorflow==2.15.0)
  - YOLOX-S 640x640
  - CPU inference

    <img width="811" alt="image" src="https://github.com/PINTO0309/onnx2tf/assets/74748700/51799aff-b006-46e1-a372-bd8b2195b854">

  Regarding on x86, AVX2 is necessary and rebuild python package in PyPI seems be NOT enabled AVX2. According to the blog, AVX2 emulation in x86 is for precision check and its is slow.

- INT8 quantization ([TexasInstruments/YOLOX-ti-lite](https://github.com/TexasInstruments/edgeai-yolox))

  In my experience, YOLOX has a very large accuracy degradation during quantization due to its structure. The reasons for this and the workaround are examined in detail by TexasInstruments. I have summarized the main points below on how to minimize accuracy degradation during quantization through my own practice. I just put into practice what TexasInstruments suggested, but the degrade in accuracy during quantization was extremely small. Note, however, that the results of the Float16 mixed-precision training before quantization are significantly degraded in accuracy due to the change in activation function to `ReLU` and many other workarounds, as well as the completely different data sets being benchmarked.

  https://github.com/PINTO0309/onnx2tf?tab=readme-ov-file#7-if-the-accuracy-of-the-int8-quantized-model-degrades-significantly

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOX-Body-Head-Hand,
    author={Katsuya Hyodo},
    title={Lightweight human detection model generated using a high-quality human dataset},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/426_YOLOX-Body-Head-Hand},
    year={2023},
    month={12},
    doi={10.5281/zenodo.10229410},
  }
  ```

## 5. Cited
  I am very grateful for their excellent work.
  - COCO-Hand

    https://vision.cs.stonybrook.edu/~supreeth/

    ```bibtex
    @article{Hand-CNN,
      title={Contextual Attention for Hand Detection in the Wild},
      author={Supreeth Narasimhaswamy and Zhengwei Wei and Yang Wang and Justin Zhang and Minh Hoai},
      booktitle={International Conference on Computer Vision (ICCV)},
      year={2019},
      url={https://arxiv.org/pdf/1904.04882.pdf}
    }
    ```
  - YOLOX

    https://github.com/Megvii-BaseDetection/YOLOX

    ```bibtex
    @article{yolox2021,
      title={YOLOX: Exceeding YOLO Series in 2021},
      author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
      journal={arXiv preprint arXiv:2107.08430},
      year={2021}
    }
    ```

  - YOLOX-ti-lite

    https://github.com/TexasInstruments/edgeai-yolox

  - yolox-ti-lite_tflite

    https://github.com/motokimura/yolox-ti-lite_tflite

  - YOLOX-Colaboratory-Training-Sample

    高橋かずひと https://github.com/Kazuhito00

    https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample


## 6. TODO
- [ ] Synthesize and retrain the dataset to further improve model performance. [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)
  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/69603b9b-ab9f-455c-a9c9-c818edc41dba)
  ```bibtex
  @INPROCEEDINGS{10323035,
    author={Beghdadi, Ayman and Beghdadi, Azeddine and Mallem, Malik and Beji, Lotfi and Cheikh, Faouzi Alaya},
    booktitle={2023 11th European Workshop on Visual Information Processing (EUVIP)},
    title={CD-COCO: A Versatile Complex Distorted COCO Database for Scene-Context-Aware Computer Vision},
    year={2023},
    volume={},
    number={},
    pages={1-6},
    doi={10.1109/EUVIP58404.2023.10323035}
  }
  ```

## 7. License
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/426_YOLOX-Body-Head-Hand/LICENSE)
