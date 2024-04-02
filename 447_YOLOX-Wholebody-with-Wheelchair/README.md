# YOLOX-Wholebody-with-Wheelchair

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 8 classes: `whole body`, `whole body with wheelchair`, `head`, `face`, `hands`, `left hand`, `right hand`, and `foot(feet)`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

Don't be ruled by the curse of mAP.

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/db7350e2-603e-42a8-8525-18a0d279d96e

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/125728c0-15fa-447f-b8ad-fb195650171b

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

- Highly accurate detection results

  **I don't evaluate by Cherry-picked data, Best-case data or Biased data at all. Therefore, only difficult images and situations in which the model is most prone to detection errors are used for validation.**

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/506e99f7-a0c2-40eb-ad80-717709a93537)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8f7b0f12-b0da-4bbd-91b9-8cdf85ade774)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/249000b4-f9e3-4311-a182-e8b9c75df7b7)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/feecbf0b-d7da-4909-ba46-cf6c1810f9b0)

## 1. Dataset
  - COCO-Hand http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)
  - I am adding my own enhancement data to COCO-Hand and re-annotating all images. In other words, only COCO images were cited and no annotation data were cited.
  - I have no plans to publish my own dataset.
  - Annotation quantity
    ```
    TOTAL: 10,578 images
    TOTAL: 254,459 labels

    train - 201,879 labels
      {
        "body": 49,413,
        "body_with_wheelchair": 580,
        "head": 42,155,
        "face": 22,680,
        "hand": 30,481,
        "hand_left": 15,257,
        "hand_right": 15,223,
        "foot": 26,090
      }

    val - 52,580 labels
      {
        "body": 13,119,
        "body_with_wheelchair": 150,
        "head": 10,839,
        "face": 5,953,
        "hand": 7,851,
        "hand_left": 3,921,
        "hand_right": 3,929,
        "foot": 6,818
      }
    ```

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8e532b3b-a00b-456e-a0c9-c162e97bf700)

  ![icon_design drawio (3)](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/72740ed3-ae9f-4ab7-9b20-bea62c58c7ac)

  |Class Name|Class ID|
  |:-|-:|
  |Body|0|
  |Body-with-Wheelchair|1|
  |Head|2|
  |Face|3|
  |Hand|4|
  |Left-Hand|5|
  |Right-Hand|6|
  |Foot (Feet)|7|

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
      demo_yolox_onnx_handLR_foot_wheelchair.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-dvw] \
      [-dwk]

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
    ```

- YOLOX-Wholebody-with-Wheelchair - Nano
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.647
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.216
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.400
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.329
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.604
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.704
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 39.275 | body_with_wheelchair | 58.763 | head      | 46.726 |
  | face       | 36.261 | hand                 | 30.635 | hand_left | 23.742 |
  | hand_right | 24.199 | foot                 | 23.180 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 48.274 | body_with_wheelchair | 69.470 | head      | 52.712 |
  | face       | 42.533 | hand                 | 41.276 | hand_left | 38.916 |
  | hand_right | 38.906 | foot                 | 36.263 |           |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.726
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.429
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.217
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.643
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.749
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 46.347 | body_with_wheelchair | 67.460 | head      | 50.705 |
  | face       | 41.743 | hand                 | 37.293 | hand_left | 32.581 |
  | hand_right | 31.823 | foot                 | 29.119 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 53.627 | body_with_wheelchair | 73.046 | head      | 55.974 |
  | face       | 47.084 | hand                 | 45.470 | hand_left | 44.720 |
  | hand_right | 43.969 | foot                 | 39.506 |           |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.761
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.491
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.620
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.235
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.493
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.552
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.704
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.826
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 53.122 | body_with_wheelchair | 73.838 | head      | 54.535 |
  | face       | 45.700 | hand                 | 43.626 | hand_left | 35.738 |
  | hand_right | 35.339 | foot                 | 35.211 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 59.120 | body_with_wheelchair | 80.066 | head      | 59.136 |
  | face       | 50.966 | hand                 | 50.113 | hand_left | 49.327 |
  | hand_right | 48.667 | foot                 | 44.281 |           |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.522
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.806
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.546
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.676
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.836
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.440
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.741
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.871
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 58.522 | body_with_wheelchair | 82.192 | head      | 56.705 |
  | face       | 48.626 | hand                 | 47.981 | hand_left | 42.161 |
  | hand_right | 41.366 | foot                 | 39.761 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 63.327 | body_with_wheelchair | 87.020 | head      | 61.033 |
  | face       | 53.470 | hand                 | 53.218 | hand_left | 52.649 |
  | hand_right | 52.140 | foot                 | 47.287 |           |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.818
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.566
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.365
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.704
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.849
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.543
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.451
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.760
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.885
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 60.562 | body_with_wheelchair | 85.181 | head      | 57.442 |
  | face       | 49.757 | hand                 | 50.008 | hand_left | 44.103 |
  | hand_right | 43.359 | foot                 | 41.618 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 64.963 | body_with_wheelchair | 88.742 | head      | 61.539 |
  | face       | 54.219 | hand                 | 54.808 | hand_left | 54.120 |
  | hand_right | 53.606 | foot                 | 48.158 |           |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.831
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.584
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.379
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.712
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.859
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.553
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.610
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.462
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.763
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.887
  per class AP:
  | class      | AP     | class                | AP     | class     | AP     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 61.485 | body_with_wheelchair | 87.158 | head      | 58.581 |
  | face       | 50.605 | hand                 | 51.344 | hand_left | 45.600 |
  | hand_right | 44.912 | foot                 | 43.187 |           |        |
  per class AR:
  | class      | AR     | class                | AR     | class     | AR     |
  |:-----------|:-------|:---------------------|:-------|:----------|:-------|
  | body       | 65.858 | body_with_wheelchair | 89.470 | head      | 62.531 |
  | face       | 54.998 | hand                 | 55.928 | hand_left | 55.133 |
  | hand_right | 54.651 | foot                 | 49.474 |           |        |
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
  @software{YOLOX-Wholebody-with-Wheelchair,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of seven classes: whole body, whole body with wheelchair, head, face, hands, left hand, right hand, and foot(feet).},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/447_YOLOX-Wholebody-with-Wheelchair},
    year={2024},
    month={4},
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
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/445_YOLOX-Body-Head-Face-HandLR-Foot-Dist/LICENSE)
