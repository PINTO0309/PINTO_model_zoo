# YOLOX-Body-Head-Hand-Face-Dist

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection model generated using a high-quality human dataset. I annotated all the data by myself. Extreme resistance to blur and occlusion. In addition, the recognition rate at short, medium, and long distances has been greatly enhanced. The camera's resistance to darkness and halation has been greatly improved.

**The use of [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco) has also greatly improved resistance to various types of noise. Compared to the [434_YOLOX-Body-Head-Hand-Face](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/434_YOLOX-Body-Head-Hand-Face) model, almost all models have a `1.0 ~ 4.0` point improvement in mAP.**

- Global distortions
  - Noise
  - Contrast
  - Compression
  - Photorealistic Rain
  - Photorealistic Haze
  - Motion-Blur
  - Defoc-Blur
  - Backlight illumination
- Local distortions
  - Motion-Blur
  - Defoc-Blur
  - Backlight illumination

|Detection results|Detection results|
|:-:|:-:|
|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/a71357c6-cdc2-4f36-a5f1-7333a1ac36c7)|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/067d9643-6411-4db3-bbe9-95a27ad682e4)|
|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/034ff9a5-a56b-4b8d-b3e6-20bab12b60e8)|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/11f727aa-2e96-4a7f-b8b7-cd2acd6f4564)|
|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/51588d5a-695f-48c8-86f4-3c222e91fe4e)|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c418a376-b448-456e-ac7f-214079b70c50)|
|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/07a55486-61cd-48b5-9c4a-0c225eef5f8c)|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/a1ba522e-a847-4288-a26a-e8194830b961)|
|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/717bcb7a-e183-43b4-8d54-23161e407884)|![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/e9bdf21e-93f7-4203-89c7-b605c7305fa1)|

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
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco) (14,667 Images, 66,903 labels, All re-annotated manually)
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
      demo_yolox_onnx_tfite.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}]
      [-dvw]

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
    ```

- YOLOX-Body-Head-Hand-Face-Dist - Nano
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.709
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.387
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.145
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.728
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 40.486 | head    | 48.518 | hand    | 31.839 |
  | face    | 37.232 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 48.910 | head    | 54.107 | hand    | 43.442 |
  | face    | 42.674 |         |        |         |        |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.760
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.450
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.602
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.156
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.663
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 45.987 | head    | 51.016 | hand    | 38.128 |
  | face    | 41.888 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 52.658 | head    | 55.809 | hand    | 46.771 |
  | face    | 47.116 |         |        |         |        |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.496
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.795
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.518
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.809
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.167
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.460
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.550
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.409
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.722
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.839
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 53.682 | head    | 55.017 | hand    | 44.393 |
  | face    | 45.503 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 58.981 | head    | 59.252 | hand    | 51.628 |
  | face    | 50.293 |         |        |         |        |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.821
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.560
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.359
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.712
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.853
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.175
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.577
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.429
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.754
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.877
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 58.626 | head    | 57.033 | hand    | 49.104 |
  | face    | 48.165 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 62.713 | head    | 60.740 | hand    | 54.724 |
  | face    | 52.555 |         |        |         |        |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.834
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.588
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.379
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.727
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.864
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.178
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.448
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.766
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.885
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 60.885 | head    | 58.342 | hand    | 51.190 |
  | face    | 49.945 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 64.634 | head    | 61.946 | hand    | 56.347 |
  | face    | 54.362 |         |        |         |        |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.568
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.837
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.611
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.396
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.746
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.882
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.181
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.457
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.781
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.901
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 62.314 | head    | 59.383 | hand    | 53.498 |
  | face    | 51.892 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | body    | 65.385 | head    | 62.703 | hand    | 57.922 |
  | face    | 56.005 |         |        |         |        |
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
  @software{YOLOX-Body-Head-Hand-Face-Dist,
    author={Katsuya Hyodo},
    title={Lightweight human detection model generated using a high-quality human dataset (Body-Head-Hand-Face) and Complex Distorted COCO database for Scene-Context-Aware computer vision},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/441_YOLOX-Body-Head-Hand-Face-Dist},
    year={2024},
    month={1},
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

  - YOLOX-ti-lite

    https://github.com/TexasInstruments/edgeai-yolox

  - yolox-ti-lite_tflite

    https://github.com/motokimura/yolox-ti-lite_tflite

  - YOLOX-Colaboratory-Training-Sample

    高橋かずひと https://github.com/Kazuhito00

    https://github.com/Kazuhito00/YOLOX-Colaboratory-Training-Sample

## 6. License
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/441_YOLOX-Body-Head-Hand-Face-Dist/LICENSE)
