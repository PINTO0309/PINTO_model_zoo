# YOLOX-Body-Head-Face-HandLR-Dist

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection model generated using a high-quality human dataset. I annotated all the data by myself. Extreme resistance to blur and occlusion. In addition, the recognition rate at short, medium, and long distances has been greatly enhanced. The camera's resistance to darkness and halation has been greatly improved.

**In addition, the ability to detect the left and right hands was added.**

The use of [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco) has also greatly improved resistance to various types of noise. Compared to the [434_YOLOX-Body-Head-Hand-Face](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/434_YOLOX-Body-Head-Hand-Face) model, almost all models have a `1.0 ~ 4.0` point improvement in mAP.

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

- The fast hand gesture motion is competitive, but the left-right decision when facing backward is quite challenging. Additional rule-based logic synthesizing face and head orientation would be needed to robustly correct the results. Alternatively, the simplest way to improve robustness would be to tie key points on the wrist output from a less accurate but very lightweight skeletal detection model.

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/2ddbe50c-ce0d-4bd9-a1f0-b6b7007ace07

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
      demo_yolox_onnx_handLR.py \
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

- YOLOX-Body-Head-Hand-Face-Dist - Nano
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.327
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.627
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.161
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.385
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.623
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 40.216 | head      | 47.832 | face       | 38.029 |
  | hand    | 31.280 | hand_left | 25.175 | hand_right | 25.566 |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 49.026 | head      | 53.489 | face       | 44.146 |
  | hand    | 41.797 | hand_left | 39.967 | hand_right | 39.787 |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.735
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.289
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.601
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.184
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.433
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.382
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.672
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.738
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 47.267 | head      | 51.593 | face       | 42.546 |
  | hand    | 38.834 | hand_left | 34.936 | hand_right | 34.202 |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 53.979 | head      | 56.300 | face       | 47.929 |
  | hand    | 46.630 | hand_left | 45.914 | hand_right | 44.710 |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.750
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.463
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.312
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.649
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.194
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.469
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.532
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.411
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.727
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 53.694 | head      | 54.517 | face       | 46.527 |
  | hand    | 44.184 | hand_left | 35.947 | hand_right | 34.897 |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 59.493 | head      | 59.105 | face       | 51.442 |
  | hand    | 50.578 | hand_left | 49.599 | hand_right | 48.779 |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - M
  ```
  ```
- YOLOX-Body-Head-Hand-Face-Dist - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.517
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.810
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.724
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.844
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.214
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.456
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.776
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.876
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 61.569 | head      | 57.917 | face       | 51.026 |
  | hand    | 51.069 | hand_left | 44.588 | hand_right | 43.741 |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 65.520 | head      | 61.566 | face       | 55.260 |
  | hand    | 55.558 | hand_left | 54.852 | hand_right | 54.406 |
  ```
- YOLOX-Body-Head-Hand-Face-Dist - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.836
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.589
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.406
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.753
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.855
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.225
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.542
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.480
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.799
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.888
  per class AP:
  | class   | AP     | class     | AP     | class      | AP     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 63.684 | head      | 59.497 | face       | 53.517 |
  | hand    | 54.638 | hand_left | 49.313 | hand_right | 48.647 |
  per class AR:
  | class   | AR     | class     | AR     | class      | AR     |
  |:--------|:-------|:----------|:-------|:-----------|:-------|
  | body    | 66.834 | head      | 62.888 | face       | 57.599 |
  | hand    | 58.656 | hand_left | 58.211 | hand_right | 57.577 |
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
  @software{YOLOX-Body-Head-Face-HandLR-Dist,
    author={Katsuya Hyodo},
    title={Lightweight human detection model generated using a high-quality human dataset (Body-Head-Face-HandLR) and Complex Distorted COCO database for Scene-Context-Aware computer vision},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/442_YOLOX-Body-Head-Face-HandLR-Dist},
    year={2024},
    month={2},
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
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/442_YOLOX-Body-Head-Face-HandLR-Dist/LICENSE)
