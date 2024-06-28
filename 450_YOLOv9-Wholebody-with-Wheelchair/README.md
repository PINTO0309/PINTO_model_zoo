# YOLOv9-Wholebody-with-Wheelchair

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 8 classes: `whole body`, `whole body with wheelchair`, `head`, `face`, `hands`, `left hand`, `right hand`, and `foot(feet)`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

Don't be ruled by the curse of mAP.

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

  - Real-time - YOLOv9-E End-to-End (Pre-process/Post-Process) ONNX + TensorRT + USB Camera

    https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/79d2cffb-9a2d-4c9a-ab92-788c8684de65

  - Still images

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
  - opencv-contrib-python 4.9.0.80+
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
      demo_yolov9_onnx_handLR_foot_wheelchair.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dlr]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for YOLOv9.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -i IMAGES_DIR, --images_dir IMAGES_DIR
        jpg, png images folder path.
      -ep {cpu,cuda,tensorrt}, \
          --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -it {fp16,int8}, --inference_type {fp16,int8}
        Inference type. Default: fp16
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

- YOLOv9-Wholebody-with-Wheelchair - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-N YOLOv9-N
                 all   2117     48988   0.353    0.352
                Body   2117     11624   0.392    0.444
  BodyWithWheelchair   2117       153   0.587    0.571
                Head   2117      9936   0.467    0.516
                Face   2117      5653   0.362    0.397
                Hand   2117      7525   0.306    0.266
           Hand-Left   2117      3739   0.237    0.198
           Hand-Right  2117      3786   0.241    0.198
                Foot   2117      6572   0.231    0.225
  ```
- YOLOv9-Wholebody-with-Wheelchair - N - ReLU (PINTO original implementation, 2.4 MB, For INT8/QAT)
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-N YOLOv9-N
                 all   2117     48988   0.353    0.351
                Body   2117     11624   0.392    0.435
  BodyWithWheelchair   2117       153   0.587    0.590
                Head   2117      9936   0.467    0.505
                Face   2117      5653   0.362    0.397
                Hand   2117      7525   0.306    0.257
           Hand-Left   2117      3739   0.237    0.200
          Hand-Right   2117      3786   0.241    0.201
                Foot   2117      6572   0.231    0.222
  ```
- YOLOv9-Wholebody-with-Wheelchair - T - Swish/SiLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-T YOLOv9-T
                 all   2117     48988   0.421    0.477
                Body   2117     11624   0.463    0.590
  BodyWithWheelchair   2117       153   0.674    0.771
                Head   2117      9936   0.507    0.591
                Face   2117      5653   0.417    0.468
                Hand   2117      7525   0.372    0.401
           Hand-Left   2117      3739   0.325    0.335
          Hand-Right   2117      3786   0.318    0.327
                Foot   2117      6572   0.291    0.332
  ```
- YOLOv9-Wholebody-with-Wheelchair - T - ReLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-T YOLOv9-T
                 all   2117     48988   0.421    0.471
                Body   2117     11624   0.463    0.567
  BodyWithWheelchair   2117       153   0.674    0.764
                Head   2117      9936   0.507    0.584
                Face   2117      5653   0.417    0.486
                Hand   2117      7525   0.372    0.390
           Hand-Left   2117      3739   0.325    0.333
          Hand-Right   2117      3786   0.318    0.323
                Foot   2117      6572   0.291    0.320
  ```
- YOLOv9-Wholebody-with-Wheelchair - S - Swish/SiLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-S
                 all   2117     48988   0.554    0.560
                Body   2117     11624   0.614    0.672
  BodyWithWheelchair   2117       153   0.871    0.844
                Head   2117      9936   0.585    0.646
                Face   2117      5653   0.506    0.555
                Hand   2117      7525   0.513    0.486
           Hand-Left   2117      3739   0.456    0.432
          Hand-Right   2117      3786   0.449    0.431
                Foot   2117      6572   0.431    0.411
  ```
- YOLOv9-Wholebody-with-Wheelchair - S - ReLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-S
                 all   2117     48988   0.554    0.556
                Body   2117     11624   0.614    0.659
  BodyWithWheelchair   2117       153   0.871    0.835
                Head   2117      9936   0.585    0.640
                Face   2117      5653   0.506    0.561
                Hand   2117      7525   0.513    0.480
           Hand-Left   2117      3739   0.456    0.430
          Hand-Right   2117      3786   0.449    0.436
                Foot   2117      6572   0.431    0.404
  ```
- YOLOv9-Wholebody-with-Wheelchair - M - Swish/SiLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-M
                 all   2117     48988   0.554    0.577
                Body   2117     11624   0.614    0.668
  BodyWithWheelchair   2117       153   0.871    0.814
                Head   2117      9936   0.585    0.666
                Face   2117      5653   0.506    0.601
                Hand   2117      7525   0.513    0.509
           Hand-Left   2117      3739   0.456    0.475
          Hand-Right   2117      3786   0.449    0.467
                Foot   2117      6572   0.431    0.417
  ```
- YOLOv9-Wholebody-with-Wheelchair - M - ReLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-M
                 all   2117     48988   0.554    0.575
                Body   2117     11624   0.614    0.663
  BodyWithWheelchair   2117       153   0.871    0.788
                Head   2117      9936   0.585    0.664
                Face   2117      5653   0.506    0.605
                Hand   2117      7525   0.513    0.513
           Hand-Left   2117      3739   0.456    0.477
          Hand-Right   2117      3786   0.449    0.475
                Foot   2117      6572   0.431    0.414
  ```
- YOLOv9-Wholebody-with-Wheelchair - C - Swish/SiLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-C
                 all   2117     48988   0.554    0.592
                Body   2117     11624   0.614    0.687
  BodyWithWheelchair   2117       153   0.871    0.822
                Head   2117      9936   0.585    0.674
                Face   2117      5653   0.506    0.608
                Hand   2117      7525   0.513    0.527
           Hand-Left   2117      3739   0.456    0.491
          Hand-Right   2117      3786   0.449    0.489
                Foot   2117      6572   0.431    0.436
  ```
- YOLOv9-Wholebody-with-Wheelchair - C - ReLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-C
                 all   2117     48988   0.554    0.593
                Body   2117     11624   0.614    0.679
  BodyWithWheelchair   2117       153   0.871    0.825
                Head   2117      9936   0.585    0.676
                Face   2117      5653   0.506    0.617
                Hand   2117      7525   0.513    0.529
           Hand-Left   2117      3739   0.456    0.491
          Hand-Right   2117      3786   0.449    0.491
                Foot   2117      6572   0.431    0.437
  ```
- YOLOv9-Wholebody-with-Wheelchair - E - Swish/SiLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-E
                 all   2117     48988   0.554    0.660
                Body   2117     11624   0.614    0.748
  BodyWithWheelchair   2117       153   0.871    0.865
                Head   2117      9936   0.585    0.723
                Face   2117      5653   0.506    0.675
                Hand   2117      7525   0.513    0.606
           Hand-Left   2117      3739   0.456    0.571
          Hand-Right   2117      3786   0.449    0.573
                Foot   2117      6572   0.431    0.523
  ```
- YOLOv9-Wholebody-with-Wheelchair - E - ReLU
  ```
               Class Images Instances     mAP50-95
                                      YOLOX-X YOLOv9-E
                 all   2117     48988   0.554    0.647
                Body   2117     11624   0.614    0.731
  BodyWithWheelchair   2117       153   0.871    0.850
                Head   2117      9936   0.585    0.719
                Face   2117      5653   0.506    0.673
                Hand   2117      7525   0.513    0.589
           Hand-Left   2117      3739   0.456    0.556
          Hand-Right   2117      3786   0.449    0.552
                Foot   2117      6572   0.431    0.508
  ```

- Pre-Process

  To ensure fair benchmark comparisons with YOLOX, `BGR to RGB conversion processing` and `normalization by division by 255.0` are added to the model input section. In addition, a `resizing process` for input images has been added to improve operational flexibility. Thus, in any model, inferences can be made at any image size. The string `1x3x{H}x{W}` at the end of the file name does not indicate the input size of the image, but the processing resolution inside the model. Therefore, the smaller the values of `{H}` and `{W}`, the lower the computational cost and the faster the inference speed. Models with larger values of `{H}` and `{W}` increase the computational cost and decrease the inference speed. Since the concept is different from the resolution of an image, any size image can be batch processed.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/ae7e86be-267e-4abf-8325-efe39add32db)

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
    --input_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --output_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --output_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression11 \
    --input_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --output_onnx_file_path yolov9_s_wholebody_with_wheelchair_post_0100_1x3x256x320.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody-with-Wheelchair,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of eight classes: whole body, whole body with wheelchair, head, face, hands, left hand, right hand, and foot(feet).},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/450_YOLOv9-Wholebody-with-Wheelchair},
    year={2024},
    month={6},
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

  - YOLOv9

    https://github.com/WongKinYiu/yolov9

    ```bibtex
    @article{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
      booktitle={arXiv preprint arXiv:2402.13616},
      year={2024}
    }
    ```

  - YOLOv9-QAT

    https://github.com/levipereira/yolov9-qat

## 6. License
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/450_YOLOv9-Wholebody-with-Wheelchair/LICENSE)
