# 468_YOLOv9-Wholebody28-Refine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `Body_with_Wheelchair`, `Body_with_Crutches`, `Head`, `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side`, `Left_Front`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Shoulder`, `Elbow`, `Hand`, `Hand_Left`, `Hand_Right`, `Knee`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

A notable feature of this model is that it can estimate the shoulder, elbow, and knee joints using only the object detection architecture. That is, I did not use any Pose Estimation architecture, nor did I use human joint keypoint data for training data. Therefore, it is now possible to estimate most of a person's parts, attributes, and keypoints through one-shot inference using a purely traditional simple object detection architecture. By not forcibly combining multiple architectures, inference performance is maximized and training costs are minimized. The difficulty of detecting the elbow is very high.

This model is transfer learning using YOLOv9-Wholebody28 weights.

Don't be ruled by the curse of mAP.

- TensorRT 10.9.0 + onnxruntime-gpu 1.22.0 + YOLOv9-Wholebody28-Refine-X

  https://github.com/user-attachments/assets/71e43ff6-7980-4134-8ad2-f82ef6babdaa

**This model, `YOLOv9-Wholebody28-Refine`, is a model that has been trained with a large number of additional annotations for objects in non-visible areas compared to `YOLOv9-Wholebody28`, thereby expanding its ability to estimate invisible parts. In particular, joint detection performance has been improved, which is a very important enhancement for acquiring skeleton detection capabilities that are planned for future support.**

|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/5f076df6-c772-4fba-be0f-2b3557477d2f)|![image](https://github.com/user-attachments/assets/61d3163b-1936-4740-991b-ecd7df141ab7)|
|![image](https://github.com/user-attachments/assets/e550120f-cf49-4b63-8728-80c72b4bf5b9)|![image](https://github.com/user-attachments/assets/52b4c72d-b8e5-441b-b4f7-f072dee9fa9d)|

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

  Halfway compromises are never acceptable. The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`.

  https://github.com/user-attachments/assets/b701dede-e5ba-4daa-8aab-f4f565be294f

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
  |Shoulder|21|Keypoints|
  |Elbow|22|Keypoints|
  |Hand|23|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|24|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|25|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Knee|26|Keypoints|
  |Foot (Feet)|27||

  ![image](https://github.com/user-attachments/assets/6e03de6c-8b81-4e3a-9a87-b863a719c37f)

## 3. Test
  - Python 3.10+
  - onnx 1.18.1+
  - onnxruntime-gpu v1.22.0 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.22.0 + TensorRT 10.9.0 + CUDA12.8 + onnx-tenosrrt-oss parser build](https://zenn.dev/pinto0309/scraps/fe82edb480254c)
  - opencv-contrib-python 4.10.0.84+
  - numpy 1.24.6+
  - TensorRT 10.9.0.34-1+cuda12.8

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
  
    **Due to a bug in the `INMSLayer` introduced in TensorRT 9.x, ONNX files containing NMS cannot be transpilated to TensorRT Engine correctly. Therefore, when running the demo program with the `-ep tensorrt` option, the demo code automatically purges all layers after NMS from the model body to avoid the TensorRT bug and perform inference correctly.**
    |Structure|Model|
    |:-:|:-:|
    |![image](https://github.com/user-attachments/assets/4b3ed411-08cb-4c32-b633-b4909dc711f2)|![image](https://github.com/user-attachments/assets/55e7bee5-f58f-408a-bafa-d542f9ec8ca2)|
    ```
    usage:
      demo_yolov9_onnx_wholebody28.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it {fp16,int8}] \
      [-dvw] \
      [-dwk] \
      [-ost OBJECT_SOCRE_THRESHOLD] \
      [-ast ATTRIBUTE_SOCRE_THRESHOLD] \
      [-kst KEYPOINT_THRESHOLD] \
      [-kdm {dot,box,both}] \
      [-dnm] \
      [-dgm] \
      [-dlr] \
      [-dhm] \
      [-drc [DISABLE_RENDER_CLASSIDS ...]] \
      [-efm] \
      [-oyt] \
      [-bblw BOUNDING_BOX_LINE_WIDTH]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for YOLOv9.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -i IMAGES_DIR, --images_dir IMAGES_DIR
        jpg, png images folder path.
      -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -it {fp16,int8}, --inference_type {fp16,int8}
        Inference type. Default: fp16
      -dvw, --disable_video_writer
        Disable video writer.
        Eliminates the file I/O load associated with automatic recording to MP4.
        Devices that use a MicroSD card or similar for main storage can speed up overall processing.
      -dwk, --disable_waitKey
        Disable cv2.waitKey().
        When you want to process a batch of still images, disable key-input wait and process them continuously.
      -ost OBJECT_SOCRE_THRESHOLD, --object_socre_threshold OBJECT_SOCRE_THRESHOLD
        The detection score threshold for object detection. Default: 0.35
      -ast ATTRIBUTE_SOCRE_THRESHOLD, --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
        The attribute score threshold for object detection. Default: 0.70
      -kst KEYPOINT_THRESHOLD, --keypoint_threshold KEYPOINT_THRESHOLD
        The keypoint score threshold for object detection. Default: 0.25
      -kdm {dot,box,both}, --keypoint_drawing_mode {dot,box,both}
        Key Point Drawing Mode. Default: dot
      -dnm, --disable_generation_identification_mode
        Disable generation identification mode. (Press N on the keyboard to switch modes)
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode. (Press G on the keyboard to switch modes)
      -dlr, --disable_left_and_right_hand_identification_mode
        Disable left and right hand identification mode. (Press H on the keyboard to switch modes)
      -dhm, --disable_headpose_identification_mode
        Disable HeadPose identification mode. (Press P on the keyboard to switch modes)
      -drc [DISABLE_RENDER_CLASSIDS ...], --disable_render_classids [DISABLE_RENDER_CLASSIDS ...]
        Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19
      -efm, --enable_face_mosaic
        Enable face mosaic.
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
      -bblw BOUNDING_BOX_LINE_WIDTH, --bounding_box_line_width BOUNDING_BOX_LINE_WIDTH
        Bounding box line width. Default: 2
    ```

<details>
<summary>YOLOv9-Wholebody28-Refine - N - Swish/SiLU (PINTO original implementation, 2.4 MB)</summary>

  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    142858 0.551 0.412 0.432    0.288
                  body   2438     12600 0.787 0.565 0.662    0.478
                 adult   2438      9595 0.715 0.576 0.619    0.471
                 child   2438      1097 0.371 0.349 0.345    0.268
                  male   2438      7183 0.637 0.614 0.615    0.472
                female   2438      2816 0.427 0.472 0.423    0.319
  body_with_wheelchair   2438       196 0.654 0.754 0.763    0.581
    body_with_crutches   2438       110 0.571 0.864 0.846    0.708
                  head   2438     10664 0.776 0.734 0.790    0.542
                 front   2438      1987 0.519 0.402 0.419    0.330
           right-front   2438      2051 0.499 0.344 0.368    0.286
            right-side   2438      1244 0.506 0.410 0.407    0.317
            right-back   2438       869 0.447 0.360 0.333    0.248
                  back   2438       519 0.302 0.218 0.157    0.116
             left-back   2438       688 0.339 0.288 0.253    0.194
             left-side   2438      1340 0.529 0.388 0.398    0.312
            left-front   2438      1966 0.482 0.347 0.375    0.295
                  face   2438      5980 0.788 0.678 0.732    0.458
                   eye   2438      5535 0.565 0.208 0.243    0.089
                  nose   2438      5221 0.603 0.318 0.363    0.166
                 mouth   2438      4195 0.549 0.271 0.293    0.113
                   ear   2438      5082 0.574 0.319 0.354    0.175
              shoulder   2438     18293 0.473 0.286 0.281    0.103
                 elbow   2438     11394 0.424 0.138 0.154    0.055
                  hand   2438      8075 0.739 0.403 0.503    0.267
             hand_left   2438      4020 0.574 0.278 0.354    0.195
            hand_right   2438      4054 0.624 0.290 0.372    0.204
                  knee   2438      9177 0.389 0.209 0.202    0.074
                  foot   2438      6907 0.568 0.445 0.462    0.238
  ```

</details>
<details>
<summary>YOLOv9-Wholebody28-Refine - T - Swish/SiLU</summary>

  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    142858 0.674 0.491 0.542    0.386
                  body   2438     12600 0.868 0.639 0.749    0.590
                 adult   2438      9595 0.809 0.651 0.724    0.599
                 child   2438      1097 0.538 0.521 0.550    0.482
                  male   2438      7183 0.730 0.672 0.721    0.599
                female   2438      2816 0.586 0.564 0.589    0.486
  body_with_wheelchair   2438       196 0.831 0.857 0.883    0.756
    body_with_crutches   2438       110 0.875 0.891 0.917    0.842
                  head   2438     10664 0.861 0.803 0.860    0.620
                 front   2438      1987 0.608 0.407 0.459    0.376
           right-front   2438      2051 0.596 0.365 0.418    0.339
            right-side   2438      1244 0.635 0.449 0.502    0.403
            right-back   2438       869 0.541 0.432 0.453    0.353
                  back   2438       519 0.383 0.279 0.244    0.190
             left-back   2438       688 0.451 0.347 0.350    0.277
             left-side   2438      1340 0.639 0.436 0.484    0.395
            left-front   2438      1966 0.585 0.380 0.425    0.350
                  face   2438      5980 0.857 0.774 0.830    0.576
                   eye   2438      5535 0.674 0.272 0.338    0.135
                  nose   2438      5221 0.732 0.424 0.495    0.252
                 mouth   2438      4195 0.666 0.350 0.402    0.173
                   ear   2438      5082 0.722 0.406 0.467    0.247
              shoulder   2438     18293 0.596 0.363 0.387    0.154
                 elbow   2438     11394 0.563 0.252 0.293    0.114
                  hand   2438      8075 0.870 0.526 0.664    0.392
             hand_left   2438      4020 0.716 0.404 0.522    0.318
            hand_right   2438      4054 0.725 0.416 0.529    0.322
                  knee   2438      9177 0.525 0.306 0.334    0.135
                  foot   2438      6907 0.691 0.548 0.596    0.338
  ```

</details>
<details>
<summary>YOLOv9-Wholebody28-Refine - S - Swish/SiLU</summary>

  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    142704 0.756 0.550 0.622    0.462
                  body   2438     13133 0.901 0.677 0.774    0.644
                 adult   2438      9934 0.879 0.670 0.776    0.683
                 child   2438      1062 0.734 0.663 0.713    0.653
                  male   2438      7074 0.828 0.717 0.790    0.689
                female   2438      2983 0.745 0.656 0.719    0.631
  body_with_wheelchair   2438       191 0.898 0.919 0.951    0.851
    body_with_crutches   2438       124 0.904 0.895 0.917    0.890
                  head   2438     11097 0.894 0.821 0.874    0.647
                 front   2438      2113 0.666 0.365 0.443    0.377
           right-front   2438      2096 0.681 0.393 0.466    0.387
            right-side   2438      1265 0.691 0.500 0.564    0.463
            right-back   2438       825 0.594 0.450 0.501    0.411
                  back   2438       633 0.561 0.306 0.369    0.289
             left-back   2438       734 0.603 0.422 0.471    0.387
             left-side   2438      1239 0.677 0.504 0.559    0.466
            left-front   2438      2190 0.673 0.349 0.440    0.371
                  face   2438      5880 0.891 0.776 0.841    0.628
                   eye   2438      5038 0.750 0.335 0.433    0.187
                  nose   2438      4818 0.821 0.505 0.595    0.329
                 mouth   2438      3898 0.761 0.435 0.519    0.249
                   ear   2438      4704 0.768 0.488 0.563    0.311
              shoulder   2438     18111 0.676 0.431 0.475    0.206
                 elbow   2438     11120 0.602 0.362 0.403    0.173
                  hand   2438      7947 0.914 0.626 0.766    0.499
             hand_left   2438      4044 0.836 0.546 0.684    0.454
            hand_right   2438      3903 0.838 0.542 0.677    0.446
                  knee   2438      9438 0.612 0.401 0.445    0.198
                  foot   2438      7110 0.764 0.633 0.689    0.421
  ```

</details>
<details>
<summary>YOLOv9-Wholebody28-Refine - C - Swish/SiLU</summary>

  ```
                   all   2438    142704 0.869 0.612 0.708    0.571
                  body   2438     13133 0.960 0.718 0.828    0.742
                 adult   2438      9934 0.943 0.701 0.828    0.774
                 child   2438      1062 0.822 0.815 0.852    0.815
                  male   2438      7074 0.935 0.779 0.861    0.802
                female   2438      2983 0.881 0.792 0.846    0.791
  body_with_wheelchair   2438       191 0.929 0.937 0.972    0.905
    body_with_crutches   2438       124 0.961 0.992 0.994    0.975
                  head   2438     11097 0.930 0.860 0.910    0.733
                 front   2438      2113 0.822 0.387 0.509    0.460
           right-front   2438      2096 0.865 0.436 0.555    0.491
            right-side   2438      1265 0.852 0.557 0.663    0.592
            right-back   2438       825 0.784 0.515 0.627    0.559
                  back   2438       633 0.845 0.385 0.523    0.455
             left-back   2438       734 0.874 0.509 0.616    0.550
             left-side   2438      1239 0.852 0.556 0.654    0.586
            left-front   2438      2190 0.839 0.382 0.515    0.462
                  face   2438      5880 0.913 0.841 0.893    0.707
                   eye   2438      5038 0.805 0.384 0.487    0.225
                  nose   2438      4818 0.900 0.551 0.647    0.388
                 mouth   2438      3898 0.832 0.491 0.584    0.308
                   ear   2438      4704 0.829 0.546 0.630    0.373
              shoulder   2438     18111 0.773 0.494 0.558    0.302
                 elbow   2438     11120 0.755 0.448 0.521    0.276
                  hand   2438      7947 0.942 0.642 0.829    0.629
             hand_left   2438      4044 0.945 0.580 0.784    0.601
            hand_right   2438      3903 0.939 0.588 0.773    0.594
                  knee   2438      9438 0.737 0.504 0.571    0.315
                  foot   2438      7110 0.859 0.735 0.803    0.566
  ```

</details>
<details>
<summary>YOLOv9-Wholebody28-Refine - E - Swish/SiLU</summary>

  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    142704 0.890 0.640 0.740    0.612
                  body   2438     13133 0.961 0.738 0.843    0.770
                 adult   2438      9934 0.957 0.719 0.846    0.806
                 child   2438      1062 0.925 0.831 0.883    0.857
                  male   2438      7074 0.951 0.786 0.866    0.819
                female   2438      2983 0.921 0.799 0.860    0.816
  body_with_wheelchair   2438       191 0.897 0.955 0.975    0.922
    body_with_crutches   2438       124 0.985 0.992 0.995    0.984
                  head   2438     11097 0.940 0.873 0.923    0.769
                 front   2438      2113 0.824 0.405 0.529    0.491
           right-front   2438      2096 0.875 0.453 0.566    0.519
            right-side   2438      1265 0.917 0.568 0.692    0.639
            right-back   2438       825 0.906 0.535 0.676    0.618
                  back   2438       633 0.838 0.409 0.550    0.494
             left-back   2438       734 0.867 0.526 0.631    0.578
             left-side   2438      1239 0.905 0.564 0.683    0.628
            left-front   2438      2190 0.859 0.402 0.545    0.501
                  face   2438      5880 0.929 0.845 0.898    0.738
                   eye   2438      5038 0.838 0.441 0.566    0.277
                  nose   2438      4818 0.916 0.594 0.703    0.448
                 mouth   2438      3898 0.855 0.551 0.648    0.363
                   ear   2438      4704 0.862 0.611 0.704    0.440
              shoulder   2438     18111 0.794 0.550 0.614    0.355
                 elbow   2438     11120 0.753 0.530 0.587    0.331
                  hand   2438      7947 0.943 0.667 0.853    0.677
             hand_left   2438      4044 0.947 0.612 0.821    0.654
            hand_right   2438      3903 0.943 0.618 0.810    0.647
                  knee   2438      9438 0.729 0.564 0.622    0.364
                  foot   2438      7110 0.873 0.785 0.842    0.624
  ```

</details>

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
    --input_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody28_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody28-Refine,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: Body, Adult, Child, Male, Female, Body_with_Wheelchair, Body_with_Crutches, Head, Front, Right_Front, Right_Side, Right_Back, Back, Left_Back, Left_Side, Left_Front, Face, Eye, Nose, Mouth, Ear, Shoulder, Elbow, Hand, Hand_Left, Hand_Right, Knee, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/468_YOLOv9-Wholebody28-Refine},
    year={2025},
    month={04},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/468_YOLOv9-Wholebody28-Refine/LICENSE)

## 7. Next Challenge
- Wrist, Hip, Ankle
- Steps and final goal

  ![image](https://github.com/user-attachments/assets/d5974b9e-018b-4739-99ae-1e5f879c0c3f)
