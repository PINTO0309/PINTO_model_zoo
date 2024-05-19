# YOLOX-Eye-Nose-Mouth-Ear

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 4 classes: `eye`, `nose`, `mouth`, `ear`. The resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

Don't be ruled by the curse of mAP.

- Sample

  https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/4ecba76a-b61d-46ca-a38c-40a905fb70c4

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/2b2c10b9-c075-4a74-b4d6-b4ebbb2a1baa)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/53ce1a12-6093-4af8-9578-539c845186d6)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/471b2994-c14c-4cd1-9a00-b68e63eb5342)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/310590a1-604c-48f6-9a82-e35195317e51)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b93a57f8-af33-4c2f-90a7-0df9d6f5114f)

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/49cfe528-c2be-422d-9632-52a98d6e1991)

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
    TOTAL: 10,883 images
    TOTAL: 88,769 labels

    train - 70,378 labels
      {
        "eye": 19379,
        "nose": 18385,
        "mouth": 14898,
        "ear": 17716
      }

    val - 18,391 labels
      {
        "eye": 4999,
        "nose": 4925,
        "mouth": 3875,
        "ear": 4592
      }
    ```

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/19c61340-0be6-48ce-a506-4209e8a45f39)

  |Class Name|Class ID|
  |:-|-:|
  |Eye|0|
  |Nose|1|
  |Mouth|2|
  |Ear|3|

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
      demo_yolox_onnx_eye_nose_mouth_ear.py \
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
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.551
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.124
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.655
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.750
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.128
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.265
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.307
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.296
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 16.543 | nose    | 24.205 | mouth   | 21.591 |
  | ear     | 22.362 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 26.926 | nose    | 33.193 | mouth   | 30.934 |
  | ear     | 31.600 |         |        |         |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - Tiny
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.609
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.147
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.685
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.822
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.140
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.288
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.315
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.724
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.833
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 18.440 | nose    | 27.151 | mouth   | 24.926 |
  | ear     | 25.282 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 27.932 | nose    | 35.623 | mouth   | 32.515 |
  | ear     | 34.093 |         |        |         |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - S
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.698
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.202
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.752
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.877
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.161
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.339
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.368
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.783
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.888
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 24.173 | nose    | 32.757 | mouth   | 30.493 |
  | ear     | 30.877 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 32.992 | nose    | 40.918 | mouth   | 37.490 |
  | ear     | 39.815 |         |        |         |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - M
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.322
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.731
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.235
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.778
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.908
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.174
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.363
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.386
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.806
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.916
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 26.831 | nose    | 35.408 | mouth   | 33.207 |
  | ear     | 33.463 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 35.060 | nose    | 42.854 | mouth   | 39.603 |
  | ear     | 41.065 |         |        |         |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - L
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.342
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.758
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.260
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.897
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.379
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.402
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.807
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.911
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 28.818 | nose    | 37.688 | mouth   | 34.504 |
  | ear     | 35.616 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 36.814 | nose    | 44.787 | mouth   | 40.686 |
  | ear     | 42.515 |         |        |         |        |
  ```
- YOLOX-Wholebody-with-Wheelchair - X
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.766
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.274
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.783
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.919
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.186
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.389
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.812
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.932
  per class AP:
  | class   | AP     | class   | AP     | class   | AP     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 30.084 | nose    | 38.311 | mouth   | 35.977 |
  | ear     | 36.761 |         |        |         |        |
  per class AR:
  | class   | AR     | class   | AR     | class   | AR     |
  |:--------|:-------|:--------|:-------|:--------|:-------|
  | eye     | 37.902 | nose    | 45.039 | mouth   | 42.007 |
  | ear     | 43.292 |         |        |         |        |
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
  @software{YOLOX-Eye-Nose-Mouth-Ear,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of four classes: eye, nose, mouth, ear.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/448_YOLOX-Eye-Nose-Mouth-Ear},
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
[Apache License Version 2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/448_YOLOX-Eye-Nose-Mouth-Ear/LICENSE)
