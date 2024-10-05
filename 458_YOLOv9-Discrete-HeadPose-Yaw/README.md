# 458_YOLOv9-Discrete-HeadPose-Yaw

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 9 classes: `Head`, `Front`, `Right-Front`, `Right-Side`, `Right-Back`, `Back`, `Left-Back`, `Left-Side`, `Left-Front`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.


This model addresses the following weaknesses of conventional HeadPose estimation models:
1. Breaks down quickly when the head is cut off outside the viewing angle
2. Pitch direction estimation is very weak
3. Estimated values ​​diverge around yaw +90° and -90°
4. Estimation accuracy is very low for yaw +90° to +180° and -90° to -180°
5. Estimation results are rough in all directions
6. Estimation is almost never successful beyond the shooting distance of 2m to 3m
7. Very vulnerable to environmental noise
8. Estimation is unstable when the depression and elevation angles of the subject and camera are large
9. Inference performance does not scale
10. Computational cost cannot be selected
11. Requires the use of fully connected layers, which are computationally very expensive

This model is transfer learning using YOLOv9-Wholebody17 weights.

Don't be ruled by the curse of mAP.

|input|output `Score threshold >= 0.35`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/23faae78-c23d-4c18-a418-8563fbf80d13)|![image](https://github.com/user-attachments/assets/ef11707f-a0f4-4561-94bd-93477841fe2a)|
|![image](https://github.com/user-attachments/assets/f83ac867-9f9f-417a-ba96-33348936e017)|![image](https://github.com/user-attachments/assets/e28b1b7f-ab75-4cc8-adf2-38e289b32818)|
|![image](https://github.com/user-attachments/assets/755b8402-7185-43b2-9201-7123d1523017)|![image](https://github.com/user-attachments/assets/f3212024-f3e3-43a4-8592-32a35427329d)|

https://github.com/user-attachments/assets/bd484aa5-681f-4075-ae6f-da464fc56c4a

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

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/user-attachments/assets/ab927e8a-c86f-4e30-b56d-c4bb6206ca91)

  ![image](https://github.com/user-attachments/assets/e1c50eb0-4da8-4f28-ba91-659ee0fa72c1)

  ![image](https://github.com/user-attachments/assets/765600a1-552d-4de9-afcc-663f6fcc1e9d) ![image](https://github.com/user-attachments/assets/15b7693a-5ffb-4c2b-9cc2-cc3022f858bb)

  |Class Name|Class ID|Remarks|
  |:-|-:|:-|
  |Head|0|Detection accuracy is higher than `Front`, `Right-Front`, `Right-Side`, `Right-Back`, `Back`, `Left-Back`, `Left-Side` and `Left-Front` bounding boxes. It is the sum of `Front`, `Right-Front`, `Right-Side`, `Right-Back`, `Back`, `Left-Back`, `Left-Side` and `Left-Front`.|
  |Front|1|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right-Front|2|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right-Side|3|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right-Back|4|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Back|5|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left-Back|6|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left-Side|7|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left-Front|8|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|

  ![image](https://github.com/user-attachments/assets/fd8d14d9-a272-45bf-b9c4-8658e5010a72)

## 3. Test
  - Python 3.10
  - onnx 1.16.1+
  - onnxruntime-gpu v1.18.1 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.18.1 + CUDA 12.5 + TensorRT 10.2.0 build (RTX3070)](https://zenn.dev/pinto0309/scraps/801db283883c38)
  - opencv-contrib-python 4.10.0.84+
  - numpy 1.24.3
  - TensorRT 10.2.0.19-1+cuda12.5

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
      demo_yolov9_discrete_head_pose_yaw.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dhp] \
      [-oyt]

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
      -dhp, --disable_headpose_identification_mode
        Disable headpose identification mode.
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
    ```

- YOLOv9-Discrete-HeadPose-Yaw - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
  WIP
  ```
- YOLOv9-Discrete-HeadPose-Yaw - T - Swish/SiLU
  ```
        Class Images Instances     P     R mAP50 mAP50-95
          all   1777     18736 0.666 0.485 0.531    0.414
         head   1777      9368 0.887 0.816 0.872    0.620
        front   1777      1911 0.670 0.436 0.498    0.402
  right-front   1777      1558 0.683 0.519 0.588    0.471
   right-side   1777      1048 0.688 0.551 0.595    0.474
   right-back   1777       688 0.625 0.442 0.483    0.370
         back   1777       467 0.470 0.278 0.280    0.214
    left-back   1777       592 0.598 0.438 0.451    0.357
    left-side   1777      1129 0.716 0.535 0.594    0.482
   left-front   1777      1975 0.658 0.350 0.416    0.336
  ```
- YOLOv9-Discrete-HeadPose-Yaw - S - Swish/SiLU
  ```
        Class Images Instances     P     R mAP50 mAP50-95
          all   1777     18736 0.755 0.520 0.600    0.487
         head   1777      9368 0.907 0.855 0.905    0.670
        front   1777      1911 0.730 0.465 0.558    0.462
  right-front   1777      1558 0.760 0.547 0.643    0.535
   right-side   1777      1048 0.767 0.588 0.663    0.551
   right-back   1777       688 0.731 0.488 0.573    0.461
         back   1777       467 0.668 0.300 0.402    0.321
    left-back   1777       592 0.689 0.498 0.533    0.441
    left-side   1777      1129 0.794 0.563 0.647    0.546
   left-front   1777      1975 0.752 0.372 0.475    0.396
  ```
- YOLOv9-Discrete-HeadPose-Yaw - C - Swish/SiLU
  ```
        Class Images Instances     P     R mAP50 mAP50-95
          all   1777     18736 0.848 0.548 0.658    0.558
         head   1777      9368 0.936 0.872 0.925    0.720
        front   1777      1911 0.813 0.477 0.601    0.518
  right-front   1777      1558 0.833 0.583 0.703    0.609
   right-side   1777      1048 0.867 0.621 0.725    0.627
   right-back   1777       688 0.836 0.526 0.650    0.548
         back   1777       467 0.792 0.367 0.476    0.400
    left-back   1777       592 0.846 0.520 0.608    0.525
    left-side   1777      1129 0.873 0.578 0.700    0.616
   left-front   1777      1975 0.837 0.385 0.537    0.464
  ```
- YOLOv9-Discrete-HeadPose-Yaw - E - Swish/SiLU
  ```
        Class Images Instances     P     R mAP50 mAP50-95
          all   1777     18736 0.872 0.557 0.678    0.593
         head   1777      9368 0.942 0.893 0.941    0.760
        front   1777      1911 0.839 0.482 0.627    0.557
  right-front   1777      1558 0.855 0.591 0.717    0.636
   right-side   1777      1048 0.868 0.622 0.737    0.659
   right-back   1777       688 0.889 0.526 0.662    0.575
         back   1777       467 0.823 0.362 0.506    0.439
    left-back   1777       592 0.871 0.547 0.636    0.564
    left-side   1777      1129 0.909 0.599 0.721    0.649
   left-front   1777      1975 0.857 0.394 0.559    0.495
  ```

- Pre-Process

  To ensure fair benchmark comparisons with YOLOX, `BGR to RGB conversion processing` and `normalization by division by 255.0` are added to the model input section. In addition, a `resizing process` for input images has been added to improve operational flexibility. Thus, in any model, inferences can be made at any image size. The string `1x3x{H}x{W}` at the end of the file name does not indicate the input size of the image, but the processing resolution inside the model. Therefore, the smaller the values of `{H}` and `{W}`, the lower the computational cost and the faster the inference speed. Models with larger values of `{H}` and `{W}` increase the computational cost and decrease the inference speed. Since the concept is different from the resolution of an image, any size image can be batch processed. e.g. 240x320, 480x640, 720x1280, ...

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
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_discrete_headpose_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Discrete-HeadPose-Yaw,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 9 classes: Head, Front, Right-Front, Right-Side, Right-Back, Back, Left-Back, Left-Side, Left-Front.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/458_YOLOv9-Discrete-HeadPose-Yaw},
    year={2024},
    month={10},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/457_YOLOv9-Wholebody17/LICENSE)

## 7. Next Challenge
- YOLOv9-Wholebody25
