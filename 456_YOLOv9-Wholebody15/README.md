# YOLOv9-Wholebody15

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 15 classes: `Body`, `Male`, `Female`, `BodyWithWheelchair`, `BodyWithCrutches`, `Head`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand-Left`, `Hand-Right`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

This model does not use facial features, but only whole-body features to estimate gender. In other words, gender can be estimated even when the body is turned backwards and the face cannot be seen at all. This model is transfer learning using YOLOv9-Wholebody13 weights.

Don't be ruled by the curse of mAP.

|Sample `Score threshold >= 0.35`|Sample `Score threshold >= 0.35`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/38b1e829-b9a4-43d6-a31f-f426995ee266)|![image](https://github.com/user-attachments/assets/284cade2-972d-4822-a274-e71217d13281)|
|![image](https://github.com/user-attachments/assets/0996bfeb-f92c-4a5e-bb0f-c2686a13b9e2)|![image](https://github.com/user-attachments/assets/39c84ea5-4edf-45e0-b5e2-6292254d483d)|
|![image](https://github.com/user-attachments/assets/f5dd22a5-6d3a-471e-acc8-cb1442d8c6b5)|![image](https://github.com/user-attachments/assets/176a2691-bf5b-42df-bcd1-556d9cfcd1d5)|

https://github.com/user-attachments/assets/68363f56-1e5f-408f-bb5d-d1a87c8f7ecf

- The `g` key on the keyboard can be used to enable or disable the gender recognition mode.
- The `h` key on the keyboard can be used to enable or disable the hand recognition mode.

  https://github.com/user-attachments/assets/479c899f-82e0-4a7c-9ca1-670ad5ed78a7

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

  ![image](https://github.com/user-attachments/assets/5d569c09-2cf3-48be-8c70-30753384b316)

  |Class Name|Class ID|Remarks|
  |:-|-:|:-|
  |Body|0|Detection accuracy is higher than `Male` and `Female` bounding boxes. It is the sum of `Male`, and `Female`.|
  |Male|1|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Female|2|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Body_with_Wheelchair|3||
  |Body_with_Crutches|4||
  |Head|5||
  |Face|6||
  |Eye|7||
  |Nose|8||
  |Mouth|9||
  |Ear|10||
  |Hand|11|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|12|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|13|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Foot (Feet)|14||

  ![image](https://github.com/user-attachments/assets/af240f2d-1459-4694-bd6e-17aa7462c5f1)

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
      demo_yolov9_onnx_wholebody15.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dlr] \
      [-dgm]

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
      -dlr, --disable_left_and_right_hand_identification_mode
        Disable left and right hand identification mode.
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode.
    ```

- YOLOv9-Wholebody15 - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.632 0.467 0.501    0.320
                  body   2384     12642 0.702 0.636 0.690    0.478
                  male   2384      7035 0.606 0.579 0.566    0.434
                female   2384      2897 0.371 0.390 0.331    0.248
  body_with_wheelchair   2384       213 0.611 0.729 0.722    0.570
    body_with_crutches   2384       112 0.522 0.848 0.775    0.616
                  head   2384     10719 0.806 0.710 0.765    0.525
                  face   2384      6152 0.810 0.632 0.689    0.428
                   eye   2384      5402 0.593 0.205 0.232   0.0881
                  nose   2384      5229 0.625 0.304 0.341    0.160
                 mouth   2384      4150 0.566 0.273 0.291    0.115
                   ear   2384      5005 0.626 0.314 0.350    0.171
                  hand   2384      8079 0.781 0.413 0.534    0.291
             hand_left   2384      4061 0.637 0.285 0.401    0.223
            hand_right   2384      4018 0.627 0.275 0.381    0.217
                  foot   2384      6909 0.596 0.411 0.451    0.233
  ```
- YOLOv9-Wholebody15 - T - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.758 0.583 0.646    0.450
                  body   2384     12642 0.793 0.756 0.808    0.615
                  male   2384      7035 0.688 0.678 0.699    0.588
                female   2384      2897 0.549 0.545 0.555    0.461
  body_with_wheelchair   2384       213 0.878 0.859 0.922    0.798
    body_with_crutches   2384       112 0.775 0.875 0.904    0.833
                  head   2384     10719 0.862 0.797 0.848    0.612
                  face   2384      6152 0.855 0.744 0.802    0.545
                   eye   2384      5402 0.709 0.286 0.342    0.135
                  nose   2384      5229 0.749 0.413 0.473    0.244
                 mouth   2384      4150 0.680 0.368 0.413    0.180
                   ear   2384      5005 0.731 0.407 0.464    0.241
                  hand   2384      8079 0.886 0.558 0.703    0.426
             hand_left   2384      4061 0.750 0.450 0.577    0.360
            hand_right   2384      4018 0.756 0.437 0.561    0.352
                  foot   2384      6909 0.704 0.571 0.620    0.352
  ```
- YOLOv9-Wholebody15 - T - ReLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.742 0.561 0.624    0.427
                  body   2384     12642 0.783 0.734 0.788    0.592
                  male   2384      7035 0.678 0.662 0.680    0.564
                female   2384      2897 0.527 0.512 0.517    0.423
  body_with_wheelchair   2384       213 0.826 0.822 0.891    0.766
    body_with_crutches   2384       112 0.723 0.875 0.883    0.793
                  head   2384     10719 0.859 0.782 0.836    0.598
                  face   2384      6152 0.846 0.736 0.794    0.530
                   eye   2384      5402 0.684 0.268 0.324    0.124
                  nose   2384      5229 0.736 0.397 0.456    0.232
                 mouth   2384      4150 0.681 0.361 0.400    0.168
                   ear   2384      5005 0.727 0.383 0.440    0.227
                  hand   2384      8079 0.887 0.523 0.679    0.401
             hand_left   2384      4061 0.744 0.414 0.549    0.335
            hand_right   2384      4018 0.734 0.403 0.527    0.323
                  foot   2384      6909 0.694 0.539 0.590    0.327
  ```
- YOLOv9-Wholebody15 - S - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.823 0.651 0.724    0.531
                  body   2384     12642 0.853 0.808 0.856    0.699
                  male   2384      7035 0.762 0.721 0.771    0.683
                female   2384      2897 0.674 0.639 0.686    0.603
  body_with_wheelchair   2384       213 0.916 0.906 0.957    0.856
    body_with_crutches   2384       112 0.850 0.911 0.919    0.886
                  head   2384     10719 0.886 0.843 0.888    0.668
                  face   2384      6152 0.893 0.779 0.833    0.623
                   eye   2384      5402 0.754 0.326 0.420    0.175
                  nose   2384      5229 0.828 0.474 0.557    0.311
                 mouth   2384      4150 0.769 0.427 0.501    0.239
                   ear   2384      5005 0.780 0.473 0.545    0.299
                  hand   2384      8079 0.918 0.660 0.794    0.525
             hand_left   2384      4061 0.855 0.569 0.708    0.479
            hand_right   2384      4018 0.837 0.566 0.701    0.469
                  foot   2384      6909 0.770 0.667 0.723    0.443
  ```
- YOLOv9-Wholebody15 - C - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.867 0.677 0.762    0.578
                  body   2384     12642 0.888 0.831 0.881    0.746
                  male   2384      7035 0.835 0.736 0.806    0.733
                female   2384      2897 0.766 0.677 0.745    0.672
  body_with_wheelchair   2384       213 0.895 0.925 0.968    0.880
    body_with_crutches   2384       112 0.922 0.902 0.928    0.908
                  head   2384     10719 0.903 0.865 0.910    0.710
                  face   2384      6152 0.919 0.825 0.875    0.680
                   eye   2384      5402 0.806 0.364 0.458    0.198
                  nose   2384      5229 0.877 0.533 0.601    0.346
                 mouth   2384      4150 0.807 0.485 0.559    0.278
                   ear   2384      5005 0.826 0.523 0.601    0.343
                  hand   2384      8079 0.929 0.631 0.821    0.585
             hand_left   2384      4061 0.910 0.571 0.754    0.544
            hand_right   2384      4018 0.889 0.564 0.744    0.538
                  foot   2384      6909 0.825 0.717 0.781    0.506
  ```
- YOLOv9-Wholebody15 - E - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2384     82623 0.891 0.725 0.813    0.638
                  body   2384     12642 0.909 0.862 0.908    0.799
                  male   2384      7035 0.871 0.757 0.841    0.783
                female   2384      2897 0.827 0.702 0.788    0.733
  body_with_wheelchair   2384       213 0.928 0.962 0.983    0.904
    body_with_crutches   2384       112 0.964 0.968 0.986    0.971
                  head   2384     10719 0.913 0.884 0.925    0.752
                  face   2384      6152 0.916 0.861 0.907    0.733
                   eye   2384      5402 0.829 0.441 0.556    0.257
                  nose   2384      5229 0.897 0.586 0.672    0.422
                 mouth   2384      4150 0.841 0.556 0.646    0.353
                   ear   2384      5005 0.854 0.593 0.676    0.409
                  hand   2384      8079 0.935 0.671 0.860    0.650
             hand_left   2384      4061 0.924 0.624 0.800    0.610
            hand_right   2384      4018 0.899 0.618 0.797    0.605
                  foot   2384      6909 0.856 0.782 0.842    0.585
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
    --input_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody15_post_0145_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody15,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 15 classes: Body, Male, Female, BodyWithWheelchair, BodyWithCrutches, Head, Face, Eye, Nose, Mouth, Ear, Hand, Hand-Left, Hand-Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/456_YOLOv9-Wholebody15},
    year={2024},
    month={8},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/456_YOLOv9-Wholebody15/LICENSE)
