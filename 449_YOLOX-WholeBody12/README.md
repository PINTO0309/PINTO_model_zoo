# YOLOX-WholeBody12

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 12 classes: `Body`, `BodyWithWheelchair`, `Head`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand-Left`, `Hand-Right`, `Foot(Feet)`. The resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

Dense features can be extracted by capturing all the key points of a face in a 2D plane, instead of capturing them in points as in RetinaFace and FaceAlignment. This is an extremely powerful capability for many tasks such as 6D Gaze Estimation, Blink Detection, FacePose, HeadPose, Gender and age estimation, facial expression estimation, and Segmentation of human body parts. It can estimate the state of a person, which cannot be accurately estimated from super sparse density information like Pose estimation. In addition, since all processing is completed by the object detection model alone, it is now possible to eliminate all cumbersome pre-processing and post-processing, as well as the pipeline of exchanging partial images that combine multiple models.

The main contributions of this model are summarized below.
- High-density information extraction
- Elimination of cumbersome pipelines
- Ultra robustness to environmental noise
- Robustness to high intensity blur due to fast camera or human body movement
- Maintains detection power in backlit or very dark environments
- Maintains detection in very bright environments
- Detection of very small objects such as 4x4 pixels
- Strong occlusion resistance

Don't be ruled by the curse of mAP.

- Sample

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3419e611-65c9-4beb-94b4-1a8788694557

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/5bfcbbf7-2e7a-4a83-a313-9190151b7507)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/34f58f96-3be9-43a1-89eb-7af587c6e538)

  ![frameE_000018](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c31ecd8c-cc73-4d22-b6cd-9c6ca6accf4d)

The use of [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco) has also greatly improved resistance to various types of noise.

- Global distortions
  - Noise
  - Contrast
  - Compression
  - Photorealistic Rain
  - Photorealistic Haze
  - Motion-Blur
  - Defocus-Blur
  - Backlight illumination
- Local distortions
  - Motion-Blur
  - Defocus-Blur
  - Backlight illumination

## 1. Dataset
  - COCO-Hand http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)
  - I am adding my own enhancement data to COCO-Hand and re-annotating all images. In other words, only COCO images were cited and no annotation data were cited.
  - I have no plans to publish my own dataset.
  - Annotation quantity
    ```
    TOTAL: 11,402 images
    TOTAL: 349,272 labels

    train - 278,548 labels
      {
        'body': 50,073,
        'body_with_wheelchair': 574,
        'head': 42,870,
        'face': 23,287,
        'eye': 20,512,
        'nose': 19,466,
        'mouth': 15,680,
        'ear': 18,768,
        'hand': 30,587,
        'hand_left': 15,342,
        'hand_right': 15,244,
        'foot': 26,145
      }

    val - 70,684 labels
      {
        'body': 13,281,
        'body_with_wheelchair': 158,
        'head': 10,942,
        'face': 6,011,
        'eye': 4,958,
        'nose': 4,709,
        'mouth': 3,802,
        'ear': 4,539,
        'hand': 7,761,
        'hand_left': 3,852,
        'hand_right': 3,908,
        'foot': 6,763
      }
    ```

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c0150a2a-7971-46c3-89f1-2d7d5aa1adf8)

  |Class Name|Class ID|
  |:-|-:|
  |Body|0|
  |BodyWithWheelchair|1|
  |Head|2|
  |Face|3|
  |Eye|4|
  |Nose|5|
  |Mouth|6|
  |Ear|7|
  |Hand|8|
  |Hand-Left|9|
  |Hand-Right|10|
  |Foot (Feet)|11|

## 3. Test
  - Python 3.10
  - onnx 1.14.1+
  - onnxruntime-gpu v1.16.1 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.16.1 + CUDA 11.8 + TensorRT 8.5.3 build (RTX3070)](https://zenn.dev/pinto0309/scraps/20afd3c58b30bf))
  - opencv-contrib-python 4.9.0.80
  - numpy 1.24.3
  - TensorRT 8.5.3-1+cuda11.8

    ```bash
    # Common ############################################
    pip install opencv-contrib-python numpy onnx

    # For ONNX ##########################################
    pip uninstall onnxruntime onnxruntime-gpu

    pip install onnxruntime
    or
    pip install onnxruntime-gpu
    ```

  - Demonstration of models with built-in post-processing (Float32/Float16)
    ```
    usage:
      demo_yolox_onnx_wholebody12.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-dvw] \
      [-dwk] \
      [-dlr]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for YOLOX.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -i IMAGES_DIR, --images_dir IMAGES_DIR
        jpg, png images folder path.
      -ep {cpu,cuda,tensorrt}, \
          --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -dvw, --disable_video_writer
        Disable video writer. Eliminates the file I/O load associated with automatic
        recording to MP4. Devices that use a MicroSD card or similar for main
        storage can speed up overall processing.
      -dwk, --disable_waitKey
        Disable cv2.waitKey(). When you want to process a batch of still images,
        disable key-input wait and process them continuously.
      -dlr, --disable_left_and_right_hand_discrimination_mode
        Disable left and right hand discrimination mode.
    ```

- YOLOX-WholeBody12 - Nano (Not usable due to missing parameters)
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.540
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.269
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.164
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.617
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.177
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.329
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.606
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.679
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 38.846 | body_with_wheelchair | 63.222 | head    | 44.729 |
  | face      | 37.625 | eye                  | 8.166  | nose    | 14.034 |
  | mouth     | 13.475 | ear                  | 17.363 | hand    | 31.303 |
  | hand_left | 24.664 | hand_right           | 25.477 | foot    | 23.695 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 47.234 | body_with_wheelchair | 71.729 | head    | 49.930 |
  | face      | 42.833 | eye                  | 10.636 | nose    | 17.912 |
  | mouth     | 20.072 | ear                  | 23.286 | hand    | 41.557 |
  | hand_left | 39.294 | hand_right           | 38.761 | foot    | 36.823 |
  ```
- YOLOX-WholeBody12 - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.339
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.327
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.573
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.199
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.291
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.653
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.729
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 45.307 | body_with_wheelchair | 68.878 | head    | 48.238 |
  | face      | 43.467 | eye                  | 12.072 | nose    | 21.591 |
  | mouth     | 18.535 | ear                  | 21.394 | hand    | 37.366 |
  | hand_left | 30.707 | hand_right           | 30.440 | foot    | 29.230 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 52.156 | body_with_wheelchair | 74.286 | head    | 53.098 |
  | face      | 48.625 | eye                  | 18.216 | nose    | 29.312 |
  | mouth     | 25.188 | ear                  | 28.364 | hand    | 45.603 |
  | hand_left | 43.241 | hand_right           | 42.523 | foot    | 40.213 |
  ```
- YOLOX-WholeBody12 - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.677
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.375
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.241
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.643
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.757
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.219
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.462
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.722
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.798
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 51.159 | body_with_wheelchair | 76.547 | head    | 51.277 |
  | face      | 47.522 | eye                  | 15.040 | nose    | 25.901 |
  | mouth     | 22.365 | ear                  | 25.307 | hand    | 43.018 |
  | hand_left | 35.301 | hand_right           | 34.820 | foot    | 34.783 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 57.031 | body_with_wheelchair | 81.955 | head    | 55.628 |
  | face      | 52.788 | eye                  | 21.344 | nose    | 33.002 |
  | mouth     | 29.084 | ear                  | 32.655 | hand    | 49.555 |
  | hand_left | 48.886 | hand_right           | 48.331 | foot    | 44.173 |
  ```
- YOLOX-WholeBody12 - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.722
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.413
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.296
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.705
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.822
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.234
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.451
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.376
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.758
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.854
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 56.493 | body_with_wheelchair | 83.737 | head    | 53.099 |
  | face      | 49.218 | eye                  | 18.310 | nose    | 29.069 |
  | mouth     | 25.380 | ear                  | 27.213 | hand    | 47.839 |
  | hand_left | 40.541 | hand_right           | 39.507 | foot    | 39.876 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 61.195 | body_with_wheelchair | 87.368 | head    | 57.166 |
  | face      | 54.021 | eye                  | 26.887 | nose    | 36.416 |
  | mouth     | 31.262 | ear                  | 34.247 | hand    | 52.943 |
  | hand_left | 52.296 | hand_right           | 50.862 | foot    | 47.789 |
  ```
- YOLOX-WholeBody12 - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.747
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.301
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.724
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.854
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.513
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.432
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.774
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.880
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 59.132 | body_with_wheelchair | 88.754 | head    | 54.140 |
  | face      | 50.880 | eye                  | 19.538 | nose    | 30.452 |
  | mouth     | 26.764 | ear                  | 28.772 | hand    | 50.160 |
  | hand_left | 44.727 | hand_right           | 43.728 | foot    | 42.114 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 63.324 | body_with_wheelchair | 91.654 | head    | 57.978 |
  | face      | 55.382 | eye                  | 28.698 | nose    | 37.486 |
  | mouth     | 32.659 | ear                  | 35.663 | hand    | 54.899 |
  | hand_left | 54.610 | hand_right           | 53.709 | foot    | 49.119 |
  ```
- YOLOX-WholeBody12 - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.760
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.452
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.736
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.858
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.247
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.477
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.432
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.785
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.883
  per class AP:
  | class     | AP     | class                | AP     | class   | AP     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 60.187 | body_with_wheelchair | 87.933 | head    | 54.923 |
  | face      | 52.507 | eye                  | 20.920 | nose    | 31.617 |
  | mouth     | 27.469 | ear                  | 29.560 | hand    | 51.390 |
  | hand_left | 46.910 | hand_right           | 45.436 | foot    | 43.374 |
  per class AR:
  | class     | AR     | class                | AR     | class   | AR     |
  |:----------|:-------|:---------------------|:-------|:--------|:-------|
  | body      | 64.089 | body_with_wheelchair | 90.902 | head    | 58.546 |
  | face      | 56.797 | eye                  | 29.920 | nose    | 38.619 |
  | mouth     | 32.957 | ear                  | 36.061 | hand    | 55.934 |
  | hand_left | 55.882 | hand_right           | 54.250 | foot    | 49.849 |
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

- INT8 quantization ([TexasInstruments/YOLOX-ti-lite](https://github.com/TexasInstruments/edgeai-yolox))

  In my experience, YOLOX has a very large accuracy degradation during quantization due to its structure. The reasons for this and the workaround are examined in detail by TexasInstruments. I have summarized the main points below on how to minimize accuracy degradation during quantization through my own practice. I just put into practice what TexasInstruments suggested, but the degrade in accuracy during quantization was extremely small. Note, however, that the results of the Float16 mixed-precision training before quantization are significantly degraded in accuracy due to the change in activation function to `ReLU` and many other workarounds, as well as the completely different data sets being benchmarked.

  https://github.com/PINTO0309/onnx2tf?tab=readme-ov-file#7-if-the-accuracy-of-the-int8-quantized-model-degrades-significantly

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOX-WholeBody12,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 12 classes: Body, BodyWithWheelchair, Head, Face, Eye, Nose, Mouth, Ear, Hand, Hand-Left, Hand-Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/449_YOLOX-WholeBody12},
    year={2024},
    month={5},
    doi={10.5281/zenodo.10229410}
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

  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)

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

## 6. License
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/449_YOLOX-WholeBody12/LICENSE)
