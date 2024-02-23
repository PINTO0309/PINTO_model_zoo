# YOLOX-Body-Head-Face-HandLR-Foot-Dist

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of seven classes: `whole body`, `head`, `face`, `hands`, `left hand`, `right hand`, and `foot(feet)`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image sets were annotated with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself.

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

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c0e3d1f-e3b2-443c-bfa5-99a7a1827dd0)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/52be36d7-7eeb-45ba-8da5-0f5782796329)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/7eca8477-8273-44b4-9569-a8d7b56ecbd5)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b2da4e8c-a999-479d-8fa8-372ff171ef1f)

- Demonstration of detection of feet wearing black socks and bare feet in all-black, hard-to-define clothing

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/1008dd5e-e939-4302-b4a1-e3438130a50b

## 1. Dataset
  - COCO-Hand http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)
  - I am adding my own enhancement data to COCO-Hand and re-annotating all images. In other words, only COCO images were cited and no annotation data were cited.
  - I have no plans to publish my own dataset.
  - Annotation quantity
    ```
    TOTAL: 10,064 images
    TOTAL: 244,388 labels

    train - 193,419 labels
      {
        "body": 47,985,
        "head": 40,422,
        "face": 21,800,
        "hand": 29,150,
        "hand_left": 14,608,
        "hand_right": 14,541,
        "foot": 24,913
      }

    val - 50,969 labels
      {
        "body": 12,831,
        "head": 11,006,
        "face": 5,771,
        "hand": 7,549,
        "hand_left": 3,790,
        "hand_right": 3,758,
        "foot": 6,264
      }
    ```

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8e532b3b-a00b-456e-a0c9-c162e97bf700)

  ![icon_design drawio (3)](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/72740ed3-ae9f-4ab7-9b20-bea62c58c7ac)

  |Class Name|Class ID|
  |:-|-:|
  |Body|0|
  |Head|1|
  |Face|2|
  |Hand|3|
  |Left-Hand|4|
  |Right-Hand|5|
  |Foot (Feet)|6|

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
      demo_yolox_onnx_handLR_foot.py \
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

- YOLOX-Body-Head-Face-HandLR-Foot-Dist - Nano
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.617
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.300
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.149
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.355
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.614
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 39.832 | head      | 46.236 | face       | 38.059 |
  | hand    | 30.681 | hand_left | 23.710 | hand_right | 24.494 |
  | foot    | 23.959 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 48.183 | head      | 51.642 | face       | 44.205 |
  | hand    | 41.074 | hand_left | 37.824 | hand_right | 38.492 |
  | foot    | 36.420 |           |        |            |        |
  ```
- YOLOX-Body-Head-Face-HandLR-Foot-Dist - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.708
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.388
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.281
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.585
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.172
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.404
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.661
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 46.426 | head      | 50.349 | face       | 42.593 |
  | hand    | 38.301 | hand_left | 33.715 | hand_right | 33.745 |
  | foot    | 30.351 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 53.056 | head      | 54.960 | face       | 48.086 |
  | hand    | 46.050 | hand_left | 44.556 | hand_right | 44.685 |
  | foot    | 40.043 |           |        |            |        |
- YOLOX-Body-Head-Face-HandLR-Foot-Dist - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.716
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.425
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.631
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.756
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.177
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.435
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.712
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 52.208 | head      | 52.599 | face       | 45.812 |
  | hand    | 43.065 | hand_left | 32.875 | hand_right | 32.383 |
  | foot    | 34.901 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 58.129 | head      | 57.021 | face       | 50.979 |
  | hand    | 49.781 | hand_left | 48.122 | hand_right | 47.827 |
  | foot    | 43.907 |           |        |            |        |
  ```
- YOLOX-Body-Head-Face-HandLR-Foot-Dist - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.774
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.343
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.690
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.834
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.195
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.471
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.435
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.749
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.867
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 57.555 | head      | 54.713 | face       | 48.840 |
  | hand    | 48.599 | hand_left | 41.270 | hand_right | 40.439 |
  | foot    | 39.804 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 62.326 | head      | 58.714 | face       | 53.595 |
  | hand    | 53.605 | hand_left | 52.609 | hand_right | 52.102 |
  | foot    | 47.186 |           |        |            |        |
  ```
- YOLOX-Body-Head-Face-HandLR-Foot-Dist - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.498
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.798
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.520
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.369
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.715
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.857
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.201
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.487
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.453
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.768
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.892
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 60.147 | head      | 55.945 | face       | 50.340 |
  | hand    | 50.840 | hand_left | 44.890 | hand_right | 44.216 |
  | foot    | 42.438 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 64.474 | head      | 59.821 | face       | 55.076 |
  | hand    | 55.525 | hand_left | 54.706 | hand_right | 54.475 |
  | foot    | 48.955 |           |        |            |        |
  ```
- YOLOX-Body-Head-Face-HandLR-Foot-Dist - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.524
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.821
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.554
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.397
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.869
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.506
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.471
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.784
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.897
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 61.893 | head      | 57.409 | face       | 52.849 |
  | hand    | 53.589 | hand_left | 48.605 | hand_right | 47.634 |
  | foot    | 44.999 |           |        |            |        |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 65.588 | head      | 60.996 | face       | 57.313 |
  | hand    | 57.589 | hand_left | 57.127 | hand_right | 56.292 |
  | foot    | 50.565 |           |        |            |        |
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
  @software{YOLOX-Body-Head-Face-HandLR-Foot-Dist,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of seven classes: whole body, head, face, hands, left hand, right hand, and foot(feet).},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/445_YOLOX-Body-Head-Face-HandLR-Foot-Dist},
    year={2024},
    month={2},
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
