# YOLOv9-Wholebody13

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 13 classes: `Body`, `BodyWithWheelchair`, `BodyWithCrutches`,`Head`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand-Left`, `Hand-Right`, `Foot(Feet)`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

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

## 1. Dataset
  - COCO-Hand http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)
  - I am adding my own enhancement data to COCO-Hand and re-annotating all images. In other words, only COCO images were cited and no annotation data were cited.
  - I have no plans to publish my own dataset.

## 2. Annotation

  Halfway compromises are never acceptable.

  ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8e532b3b-a00b-456e-a0c9-c162e97bf700)

  ![icon_design drawio (3)](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/72740ed3-ae9f-4ab7-9b20-bea62c58c7ac)

  |Class Name|Class ID|
  |:-|-:|
  |Body|0|
  |BodyWithWheelchair|1|
  |BodyWithCrutches|2|
  |Head|3|
  |Face|4|
  |Eye|5|
  |Nose|6|
  |Mouth|7|
  |Ear|8|
  |Hand|9|
  |Hand-Left|10|
  |Hand-Right|11|
  |Foot (Feet)|12|

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
      demo_yolov9_onnx_wholebody13.py \
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

- YOLOv9-Wholebody13 - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.678 0.474 0.528    0.333
                  body   2385     12199 0.714 0.630 0.691    0.469
  body_with_wheelchair   2385       182 0.696 0.824 0.845    0.673
    body_with_crutches   2385       103 0.597 0.893 0.892    0.743
                  head   2385     10343 0.805 0.718 0.778    0.526
                  face   2385      5561 0.846 0.638 0.699    0.446
                   eye   2385      5211 0.567 0.198 0.216    0.079
                  nose   2385      4818 0.666 0.298 0.339    0.158
                 mouth   2385      3936 0.602 0.282 0.302    0.118
                   ear   2385      4874 0.683 0.303 0.347    0.169
                  hand   2385      7791 0.777 0.406 0.531    0.286
             hand_left   2385      3938 0.638 0.274 0.394    0.216
            hand_right   2385      3853 0.611 0.268 0.367    0.207
                  foot   2385      6782 0.613 0.433 0.468    0.240
  ```
- YOLOv9-Wholebody13 - N - ReLU (PINTO original implementation, 2.4 MB, For INT8/QAT)
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.675  0.47 0.523    0.329
                  body   2385     12199 0.687 0.627 0.674    0.449
  body_with_wheelchair   2385       182 0.671 0.868 0.870    0.682
    body_with_crutches   2385       103 0.559 0.883 0.861    0.747
                  head   2385     10343 0.787 0.702 0.758    0.511
                  face   2385      5561 0.825 0.637 0.696    0.442
                   eye   2385      5211 0.613 0.201 0.232    0.083
                  nose   2385      4818 0.681 0.306 0.343    0.160
                 mouth   2385      3936 0.637  0.28 0.310    0.118
                   ear   2385      4874 0.671 0.302 0.346    0.166
                  hand   2385      7791 0.773 0.388 0.516    0.276
             hand_left   2385      3938 0.626 0.263 0.383    0.209
            hand_right   2385      3853 0.635 0.249 0.361    0.200
                  foot   2385      6782 0.615 0.403 0.446    0.227
  ```
- YOLOv9-Wholebody13 - T - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.772 0.583 0.652    0.442
                  body   2385     12199 0.796 0.749 0.803    0.603
  body_with_wheelchair   2385       182 0.835 0.923 0.950    0.845
    body_with_crutches   2385       103 0.694 0.913 0.936    0.867
                  head   2385     10343 0.855 0.794 0.850    0.609
                  face   2385      5561 0.861 0.755 0.812    0.559
                   eye   2385      5211 0.682 0.270 0.317    0.125
                  nose   2385      4818 0.759 0.430 0.489    0.244
                 mouth   2385      3936 0.705 0.373 0.412    0.179
                   ear   2385      4874 0.753 0.399 0.454    0.239
                  hand   2385      7791 0.888 0.545 0.697    0.420
             hand_left   2385      3938 0.736 0.446 0.579    0.355
            hand_right   2385      3853 0.748 0.419 0.551    0.344
                  foot   2385      6782 0.730 0.558 0.622    0.354
  ```
- YOLOv9-Wholebody13 - S - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.838 0.653 0.726    0.514
                  body   2385     12199 0.835 0.805 0.856    0.683
  body_with_wheelchair   2385       182 0.883 0.951 0.980    0.889
    body_with_crutches   2385       103 0.912 0.932 0.956    0.922
                  head   2385     10343 0.879 0.846 0.891    0.660
                  face   2385      5561 0.894 0.784 0.837    0.625
                   eye   2385      5211 0.750 0.327 0.406    0.169
                  nose   2385      4818 0.820 0.491 0.567    0.310
                 mouth   2385      3936 0.765 0.424 0.494    0.237
                   ear   2385      4874 0.790 0.476 0.539    0.300
                  hand   2385      7791 0.913 0.657 0.795    0.516
             hand_left   2385      3938 0.828 0.577 0.706    0.466
            hand_right   2385      3853 0.833 0.554 0.689    0.455
                  foot   2385      6782 0.789 0.662 0.729    0.446
  ```
- YOLOv9-Wholebody13 - M - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.848 0.674 0.753    0.540
                  body   2385     12199 0.835 0.820 0.865    0.697
  body_with_wheelchair   2385       182 0.900 0.934 0.971    0.888
    body_with_crutches   2385       103 0.890 0.947 0.975    0.928
                  head   2385     10343 0.874 0.863 0.903    0.683
                  face   2385      5561 0.896 0.836 0.878    0.670
                   eye   2385      5211 0.769 0.342 0.435    0.186
                  nose   2385      4818 0.849 0.532 0.616    0.351
                 mouth   2385      3936 0.805 0.459 0.540    0.269
                   ear   2385      4874 0.808 0.494 0.570    0.322
                  hand   2385      7791 0.909 0.667 0.817    0.550
             hand_left   2385      3938 0.861 0.590 0.742    0.510
            hand_right   2385      3853 0.853 0.583 0.730    0.498
                  foot   2385      6782 0.779 0.696 0.751    0.471
  ```
- YOLOv9-Wholebody13 - C - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.861 0.676 0.761    0.553
                  body   2385     12199 0.850 0.832 0.874    0.716
  body_with_wheelchair   2385       182 0.898 0.934 0.968    0.892
    body_with_crutches   2385       103 0.967 0.942 0.974    0.937
                  head   2385     10343 0.880 0.870 0.909    0.694
                  face   2385      5561 0.900 0.833 0.875    0.676
                   eye   2385      5211 0.778 0.347 0.452    0.197
                  nose   2385      4818 0.856 0.537 0.624    0.363
                 mouth   2385      3936 0.810 0.468 0.553    0.277
                   ear   2385      4874 0.802 0.505 0.576    0.331
                  hand   2385      7791 0.909 0.646 0.820    0.569
             hand_left   2385      3938 0.869 0.581 0.749    0.524
            hand_right   2385      3853 0.870 0.570 0.736    0.516
                  foot   2385      6782 0.805 0.720 0.777    0.499
  ```
- YOLOv9-Wholebody13 - E - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2385     69591 0.886 0.723 0.806    0.607
                  body   2385     12199 0.890 0.859 0.904    0.773
  body_with_wheelchair   2385       182 0.924 0.951 0.974    0.911
    body_with_crutches   2385       103 0.971 0.969 0.991    0.962
                  head   2385     10343 0.898 0.888 0.926    0.735
                  face   2385      5561 0.897 0.874 0.907    0.727
                   eye   2385      5211 0.807 0.438 0.540    0.251
                  nose   2385      4818 0.894 0.607 0.690    0.423
                 mouth   2385      3936 0.835 0.551 0.632    0.345
                   ear   2385      4874 0.839 0.584 0.662    0.403
                  hand   2385      7791 0.930 0.663 0.853    0.627
             hand_left   2385      3938 0.899 0.622 0.789    0.587
            hand_right   2385      3853 0.882  0.61 0.780    0.576
                  foot   2385      6782 0.850 0.778 0.837    0.574
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
    --input_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_wholebody13_post_0245_1x3x544x960.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody13,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 13 classes: Body, BodyWithWheelchair, BodyWithCrutches, Head, Face, Eye, Nose, Mouth, Ear, Hand, Hand-Left, Hand-Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/454_YOLOv9-Wholebody13},
    year={2024},
    month={7},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/454_YOLOv9-Wholebody13/LICENSE)
