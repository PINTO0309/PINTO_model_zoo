# 459_YOLOv9-Wholebody25

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 25 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `Body_with_Wheelchair`, `Body_with_Crutches`, `Head`, `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side`, `Left_Front`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand_Left`, `Hand_Right`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

The aim is to estimate head pose direction with minimal computational cost using only an object detection model, with an emphasis on practical aspects. The concept is significantly different from existing full-mesh type head direction estimation models, head direction estimation models with tweaked loss functions, and models that perform precise 360° 6D estimation. Capturing the features of every part of the body on a 2D surface makes it very easy to combine with other feature extraction processes. In experimental trials, the model was trained to only estimate eight Yaw directions, but I plan to add the ability to estimate five Pitch directions in the future.

This model is transfer learning using YOLOv9-Wholebody17 weights.

Don't be ruled by the curse of mAP.

https://github.com/user-attachments/assets/c8f5fb1a-e411-4d2b-a080-4cfc4bcce5af

|input|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/dd0027dd-c998-4297-9efe-4aae802a7783)|![image](https://github.com/user-attachments/assets/5adf1fc0-4a96-40ee-b241-469b71fbb867)|
|![image](https://github.com/user-attachments/assets/436a2cfa-776d-48b8-93c7-90a1dbb9da29)|![image](https://github.com/user-attachments/assets/9e6dd578-7ca2-4903-bbec-9f0f1b1ca583)|
|![image](https://github.com/user-attachments/assets/5e3372ec-e80d-40fe-8378-fbfc2a9c1a8a)|![image](https://github.com/user-attachments/assets/e7fddd99-d424-47d4-9526-0cbdcc07d18e)|
|![image](https://github.com/user-attachments/assets/1fa6132b-26b1-4748-8640-3bd7c5aa057e)|![image](https://github.com/user-attachments/assets/91dd67ff-2ded-460a-868f-eca0a1bcb8d9)|
|![image](https://github.com/user-attachments/assets/1bd51ec6-775c-4f35-89a7-be055af03609)|![image](https://github.com/user-attachments/assets/208fe0c0-4a48-44f6-8f77-ec1d35e024ba)|
|![image](https://github.com/user-attachments/assets/54d9547c-6e6b-4f78-a244-2cd257a2cf6b)|![image](https://github.com/user-attachments/assets/2a75b535-315f-4406-90a4-6131fff009a8)|

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

  Halfway compromises are never acceptable. I added `2,611` annotations to the following `480x360` image. The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`.

  ![image](https://github.com/user-attachments/assets/ca0b0b44-4280-49aa-b257-fca8429b3337)

  Please feel free to change the head direction label as you wish. There is no correlation between the model's behavior and the meaning of the label text.

  ![image](https://github.com/user-attachments/assets/765600a1-552d-4de9-afcc-663f6fcc1e9d) ![image](https://github.com/user-attachments/assets/15b7693a-5ffb-4c2b-9cc2-cc3022f858bb)

  |Class Name|Class ID|Remarks|
  |:-|-:|:-|
  |Body|0|Detection accuracy is higher than `Adult`, `Child`, `Male` and `Female` bounding boxes. It is the sum of `Adult`, `Child`, `Male`, and `Female`.|
  |Adult|1|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Child|2|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Male|3|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Female|4|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Body_with_Wheelchair|5||
  |Body_with_Crutches|6||
  |Head|7|Detection accuracy is higher than `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side` and `Left_Front` bounding boxes. It is the sum of `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side` and `Left_Front`.|
  |Front|8|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right_Front|9|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right_Side|10|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Right_Back|11|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Back|12|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left_Back|13|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left_Side|14|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Left_Front|15|Bounding box coordinates are shared with `Head`. It is defined as a subclass of `Head` as a superclass.|
  |Face|16||
  |Eye|17||
  |Nose|18||
  |Mouth|19||
  |Ear|20||
  |Hand|21|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|22|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|23|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Foot (Feet)|24||

  ![image](https://github.com/user-attachments/assets/49f9cbf3-3a9c-4666-84ae-d86148c34866)

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
      demo_yolov9_onnx_wholebody25.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dnm] \
      [-dgm] \
      [-dlr] \
      [-dhm] \
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
      -dnm, --disable_generation_identification_mode
        Disable generation identification mode.
        (Press N on the keyboard to switch modes)
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode.
        (Press G on the keyboard to switch modes)
      -dlr, --disable_left_and_right_hand_identification_mode
        Disable left and right hand identification mode.
        (Press H on the keyboard to switch modes)
      -dhm, --disable_headpose_identification_mode
        Disable HeadPose identification mode.
        (Press P on the keyboard to switch modes)
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
    ```

- YOLOv9-Wholebody25 - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
                Class Images Instances     P     R mAP50 mAP50-95
                  all   2438    103994 0.586 0.430 0.451    0.307
                 body   2438     12600 0.797 0.549 0.654    0.470
                adult   2438      9595 0.717 0.586 0.618    0.463
                child   2438      1097 0.409 0.348 0.359    0.278
                 male   2438      7183 0.638 0.624 0.624    0.469
               female   2438      2816 0.429 0.451 0.421    0.316
  body_with_wheelchai   2438       196 0.658 0.718 0.749    0.583
   body_with_crutches   2438       110 0.647 0.882 0.851    0.714
                 head   2438     10664 0.787 0.741 0.792    0.536
                front   2438      1987 0.526 0.387 0.397    0.307
          right-front   2438      2051 0.506 0.340 0.355    0.275
           right-side   2438      1244 0.518 0.398 0.379    0.292
           right-back   2438       869 0.465 0.366 0.325    0.239
                 back   2438       519 0.274 0.197 0.131   0.0986
            left-back   2438       688 0.368 0.294 0.239    0.181
            left-side   2438      1340 0.533 0.378 0.380    0.294
           left-front   2438      1966 0.489 0.341 0.359    0.282
                 face   2438      5980 0.819 0.657 0.706    0.442
                  eye   2438      5535 0.589 0.204 0.228   0.0858
                 nose   2438      5221 0.640 0.310 0.347    0.160
                mouth   2438      4195 0.592 0.277 0.297    0.112
                  ear   2438      5082 0.635 0.318 0.353    0.173
                 hand   2438      8075 0.768 0.398 0.516    0.272
            hand_left   2438      4020 0.618 0.264 0.360    0.196
           hand_right   2438      4054 0.636 0.276 0.375    0.206
                 foot   2438      6907 0.596 0.438 0.462    0.236
  ```
- YOLOv9-Wholebody25 - T - Swish/SiLU
  ```
                Class Images Instances     P     R mAP50 mAP50-95
                  all   2438    103994 0.685 0.518 0.566    0.415
                 body   2438     12600 0.876 0.657 0.761    0.597
                adult   2438      9595 0.803 0.676 0.734    0.604
                child   2438      1097 0.528 0.514 0.540    0.475
                 male   2438      7183 0.729 0.702 0.742    0.610
               female   2438      2816 0.580 0.594 0.614    0.505
  body_with_wheelchai   2438       196 0.870 0.854 0.898    0.773
   body_with_crutches   2438       110 0.805 0.882 0.919    0.850
                 head   2438     10664 0.847 0.819 0.866    0.616
                front   2438      1987 0.587 0.414 0.450    0.365
          right-front   2438      2051 0.582 0.379 0.419    0.336
           right-side   2438      1244 0.603 0.458 0.483    0.388
           right-back   2438       869 0.535 0.421 0.432    0.336
                 back   2438       519 0.362 0.260 0.214    0.167
            left-back   2438       688 0.446 0.352 0.340    0.265
            left-side   2438      1340 0.625 0.437 0.474    0.381
           left-front   2438      1966 0.593 0.380 0.426    0.346
                 face   2438      5980 0.853 0.777 0.822    0.563
                  eye   2438      5535 0.682 0.270 0.331    0.131
                 nose   2438      5221 0.757 0.416 0.476    0.239
                mouth   2438      4195 0.687 0.355 0.403    0.173
                  ear   2438      5082 0.728 0.405 0.456    0.242
                 hand   2438      8075 0.869 0.532 0.670    0.401
            hand_left   2438      4020 0.735 0.408 0.536    0.328
           hand_right   2438      4054 0.744 0.422 0.539    0.332
                 foot   2438      6907 0.704 0.557 0.607    0.342
  ```
- YOLOv9-Wholebody25 - S - Swish/SiLU
  ```
                Class Images Instances     P     R mAP50 mAP50-95
                  all   2438    103994 0.755 0.588 0.648    0.498
                 body   2438     12600 0.895 0.728 0.821    0.684
                adult   2438      9595 0.872 0.727 0.810    0.708
                child   2438      1097 0.711 0.659 0.726    0.667
                 male   2438      7183 0.816 0.772 0.828    0.719
               female   2438      2816 0.732 0.700 0.754    0.663
  body_with_wheelchai   2438       196 0.885 0.946 0.958    0.860
   body_with_crutches   2438       110 0.913 0.918 0.948    0.904
                 head   2438     10664 0.889 0.859 0.905    0.673
                front   2438      1987 0.650 0.447 0.501    0.419
          right-front   2438      2051 0.648 0.423 0.478    0.392
           right-side   2438      1244 0.655 0.515 0.556    0.457
           right-back   2438       869 0.607 0.483 0.512    0.416
                 back   2438       519 0.454 0.324 0.299    0.241
            left-back   2438       688 0.527 0.407 0.415    0.335
            left-side   2438      1340 0.671 0.486 0.537    0.448
           left-front   2438      1966 0.621 0.419 0.471    0.395
                 face   2438      5980 0.900 0.793 0.848    0.634
                  eye   2438      5535 0.734 0.317 0.402    0.171
                 nose   2438      5221 0.819 0.484 0.563    0.310
                mouth   2438      4195 0.754 0.425 0.491    0.231
                  ear   2438      5082 0.777 0.479 0.547    0.303
                 hand   2438      8075 0.909 0.637 0.767    0.500
            hand_left   2438      4020 0.829 0.550 0.673    0.446
           hand_right   2438      4054 0.832 0.545 0.674    0.448
                 foot   2438      6907 0.767 0.658 0.712    0.437
  ```
- YOLOv9-Wholebody25 - C - Swish/SiLU
  ```
                Class Images Instances     P     R mAP50 mAP50-95
                  all   2438    103994 0.819 0.618 0.701    0.562
                 body   2438     12600 0.920 0.761 0.860    0.749
                adult   2438      9595 0.919 0.746 0.851    0.775
                child   2438      1097 0.866 0.737 0.824    0.778
                 male   2438      7183 0.892 0.799 0.868    0.787
               female   2438      2816 0.858 0.757 0.834    0.759
  body_with_wheelchai   2438       196 0.893 0.918 0.948    0.871
   body_with_crutches   2438       110 0.963 0.927 0.983    0.949
                 head   2438     10664 0.926 0.883 0.929    0.722
                front   2438      1987 0.709 0.473 0.551    0.474
          right-front   2438      2051 0.732 0.435 0.526    0.444
           right-side   2438      1244 0.730 0.547 0.608    0.518
           right-back   2438       869 0.668 0.501 0.573    0.479
                 back   2438       519 0.553 0.337 0.362    0.296
            left-back   2438       688 0.647 0.441 0.490    0.410
            left-side   2438      1340 0.758 0.512 0.589    0.507
           left-front   2438      1966 0.704 0.443 0.524    0.453
                 face   2438      5980 0.915 0.858 0.903    0.705
                  eye   2438      5535 0.770 0.363 0.456    0.205
                 nose   2438      5221 0.869 0.542 0.625    0.361
                mouth   2438      4195 0.793 0.489 0.562    0.280
                  ear   2438      5082 0.814 0.532 0.609    0.355
                 hand   2438      8075 0.932 0.620 0.804    0.577
            hand_left   2438      4020 0.896 0.557 0.733    0.531
           hand_right   2438      4054 0.907 0.560 0.730    0.533
                 foot   2438      6907 0.842 0.719 0.785    0.524
  ```
- YOLOv9-Wholebody25 - E - Swish/SiLU
  ```
                Class Images Instances     P     R mAP50 mAP50-95
                  all   2438    103994 0.846 0.649 0.737    0.607
                 body   2438     12600 0.933 0.794 0.886    0.792
                adult   2438      9595 0.942 0.761 0.876    0.816
                child   2438      1097 0.880 0.769 0.847    0.812
                 male   2438      7183 0.920 0.816 0.888    0.823
               female   2438      2816 0.886 0.778 0.854    0.794
  body_with_wheelchai   2438       196 0.895 0.949 0.978    0.905
   body_with_crutches   2438       110 0.963 0.945 0.985    0.965
                 head   2438     10664 0.929 0.904 0.942    0.758
                front   2438      1987 0.722 0.484 0.578    0.510
          right-front   2438      2051 0.744 0.449 0.543    0.473
           right-side   2438      1244 0.772 0.557 0.634    0.557
           right-back   2438       869 0.757 0.544 0.618    0.527
                 back   2438       519 0.650 0.372 0.435    0.374
            left-back   2438       688 0.710 0.471 0.524    0.455
            left-side   2438      1340 0.763 0.528 0.609    0.539
           left-front   2438      1966 0.734 0.460 0.553    0.491
                 face   2438      5980 0.916 0.886 0.923    0.748
                  eye   2438      5535 0.831 0.429 0.544    0.255
                 nose   2438      5221 0.895 0.579 0.669    0.413
                mouth   2438      4195 0.832 0.545 0.635    0.341
                  ear   2438      5082 0.860 0.589 0.672    0.416
                 hand   2438      8075 0.932 0.657 0.840    0.638
            hand_left   2438      4020 0.914 0.600 0.773    0.591
           hand_right   2438      4054 0.916 0.596 0.777    0.594
                 foot   2438      6907 0.863 0.773 0.835    0.590
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
    --input_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody25_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody25,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 25 classes: Body, Adult, Child, Male, Female, Body_with_Wheelchair, Body_with_Crutches, Head, Front, Right_Front, Right_Side, Right_Back, Back, Left_Back, Left_Side, Left_Front, Face, Eye, Nose, Mouth, Ear, Hand, Hand_Left, Hand_Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/459_YOLOv9-Wholebody25/LICENSE)

## 7. Next Challenge
- Pitch x5 classes

## 8. Practical application
1. https://github.com/PINTO0309/yolov9-wholebody25-tensorflowjs-web-test
