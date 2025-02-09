# 464_YOLOv9-Wholebody28

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `Body_with_Wheelchair`, `Body_with_Crutches`, `Head`, `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side`, `Left_Front`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Shoulder`, `Elbow`, `Hand`, `Hand_Left`, `Hand_Right`, `Knee`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

This model is transfer learning using YOLOv9-Wholebody25 weights.

Don't be ruled by the curse of mAP.

https://github.com/user-attachments/assets/7d5ceafb-12c3-476f-8077-f11c54b1de52

|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|
|:-:|:-:|
|![000000003786](https://github.com/user-attachments/assets/a21ae3c3-4bea-4461-9ad3-ea1decb14f63)|![000000010082](https://github.com/user-attachments/assets/2654eac6-219e-4840-adad-5a1d24629b56)|
|![000000061606](https://github.com/user-attachments/assets/9348193d-c871-445b-b19d-aa6244d5a543)|![000000064744](https://github.com/user-attachments/assets/c9a2935b-a7f8-4172-947c-fd158b35491e)|
|![000000088214](https://github.com/user-attachments/assets/8b597dc4-9bc9-454a-96fc-c61914958097)|![000000088754](https://github.com/user-attachments/assets/def8dba4-7109-4908-a67a-f33053e585ce)|
|![frameE_000031](https://github.com/user-attachments/assets/d6a774ae-b44a-43a4-bf78-034ec1522492)|![frameE_000071](https://github.com/user-attachments/assets/15b19dbc-bcaf-4b04-9262-38c4f08f78a6)|

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

  ![image](https://github.com/user-attachments/assets/651764ae-7300-431d-8bb2-0a1f61ebac63)

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

- YOLOv9-Wholebody28 - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
```
```
- YOLOv9-Wholebody28 - T - Swish/SiLU
```
```
- YOLOv9-Wholebody28 - S - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.752 0.563 0.633    0.465
                  body   2438     13133 0.894 0.744 0.832    0.671
                 adult   2438      9934 0.879 0.683 0.797    0.688
                 child   2438      1062 0.751 0.690 0.750    0.686
                  male   2438      7074 0.826 0.769 0.841    0.716
                female   2438      2983 0.775 0.690 0.768    0.662
  body_with_wheelchair   2438       191 0.892 0.906 0.957    0.846
    body_with_crutches   2438       124 0.945 0.895 0.922    0.901
                  head   2438     11097 0.861 0.859 0.897    0.650
                 front   2438      2113 0.653 0.376 0.461    0.381
           right-front   2438      2096 0.683 0.428 0.512    0.415
            right-side   2438      1265 0.675 0.508 0.573    0.463
            right-back   2438       825 0.598 0.463 0.507    0.406
                  back   2438       633 0.564 0.324 0.375    0.286
             left-back   2438       734 0.612 0.429 0.483    0.389
             left-side   2438      1239 0.675 0.507 0.573    0.472
            left-front   2438      2190 0.680 0.356 0.449    0.369
                  face   2438      5880 0.886 0.786 0.833    0.615
                   eye   2438      5038 0.745 0.337 0.417    0.176
                  nose   2438      4818 0.807 0.509 0.582    0.316
                 mouth   2438      3898 0.761 0.437 0.502    0.237
                   ear   2438      4704 0.770 0.491 0.551    0.303
              shoulder   2438     17126 0.635 0.449 0.474    0.195
                 elbow   2438     10464 0.572 0.342 0.372    0.149
                  hand   2438      7947 0.912 0.652 0.782    0.507
             hand_left   2438      4044 0.849 0.559 0.706    0.464
            hand_right   2438      3903 0.836 0.559 0.694    0.457
                  knee   2438      8756 0.566 0.381 0.410    0.175
                  foot   2438      7110 0.765 0.648 0.701    0.428
  ```
- YOLOv9-Wholebody28 - S - ReLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.728 0.548 0.611    0.444
                  body   2438     13133 0.883 0.730 0.821    0.647
                 adult   2438      9934 0.857 0.679 0.781    0.661
                 child   2438      1062 0.689 0.632 0.704    0.635
                  male   2438      7074 0.791 0.754 0.817    0.684
                female   2438      2983 0.716 0.665 0.732    0.622
  body_with_wheelchair   2438       191 0.884 0.927 0.956    0.859
    body_with_crutches   2438       124 0.924 0.887 0.920    0.892
                  head   2438     11097 0.845 0.849 0.885    0.630
                 front   2438      2113 0.650 0.374 0.451    0.369
           right-front   2438      2096 0.664 0.424 0.498    0.400
            right-side   2438      1265 0.655 0.499 0.555    0.440
            right-back   2438       825 0.572 0.456 0.483    0.385
                  back   2438       633 0.531 0.316 0.352    0.265
             left-back   2438       734 0.592 0.421 0.458    0.363
             left-side   2438      1239 0.667 0.496 0.556    0.455
            left-front   2438      2190 0.645 0.350 0.431    0.353
                  face   2438      5880 0.875 0.774 0.821    0.599
                   eye   2438      5038 0.722 0.321 0.396    0.167
                  nose   2438      4818 0.788 0.488 0.555    0.297
                 mouth   2438      3898 0.743 0.425 0.480    0.219
                   ear   2438      4704 0.744 0.475 0.531    0.287
              shoulder   2438     17126 0.605 0.432 0.449    0.180
                 elbow   2438     10464 0.540 0.321 0.344    0.134
                  hand   2438      7947 0.896 0.627 0.757    0.475
             hand_left   2438      4044 0.813 0.527 0.669    0.428
            hand_right   2438      3903 0.809 0.532 0.662    0.423
                  knee   2438      8756 0.541 0.352 0.378    0.154
                  foot   2438      7110 0.740 0.621 0.673    0.402
  ```
- YOLOv9-Wholebody28 - C - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.829 0.608 0.705    0.541
                  body   2438     13133 0.913 0.782 0.869    0.734
                 adult   2438      9934 0.923 0.708 0.846    0.756
                 child   2438      1062 0.863 0.769 0.839    0.786
                  male   2438      7074 0.891 0.807 0.887    0.787
                female   2438      2983 0.870 0.763 0.848    0.760
  body_with_wheelchair   2438       191 0.900 0.948 0.970    0.874
    body_with_crutches   2438       124 0.981 0.911 0.949    0.931
                  head   2438     11097 0.892 0.899 0.934    0.712
                 front   2438      2113 0.761 0.397 0.530    0.451
           right-front   2438      2096 0.784 0.457 0.602    0.502
            right-side   2438      1265 0.798 0.542 0.663    0.555
            right-back   2438       825 0.787 0.502 0.644    0.537
                  back   2438       633 0.746 0.377 0.511    0.409
             left-back   2438       734 0.813 0.488 0.606    0.513
             left-side   2438      1239 0.801 0.544 0.657    0.560
            left-front   2438      2190 0.785 0.379 0.531    0.447
                  face   2438      5880 0.902 0.856 0.899    0.693
                   eye   2438      5038 0.775 0.396 0.489    0.219
                  nose   2438      4818 0.861 0.576 0.650    0.381
                 mouth   2438      3898 0.798 0.511 0.583    0.296
                   ear   2438      4704 0.803 0.549 0.622    0.361
              shoulder   2438     17126 0.709 0.502 0.547    0.257
                 elbow   2438     10464 0.648 0.413 0.454    0.205
                  hand   2438      7947 0.928 0.632 0.816    0.582
             hand_left   2438      4044 0.911 0.566 0.767    0.553
            hand_right   2438      3903 0.899 0.583 0.761    0.548
                  knee   2438      8756 0.651 0.453 0.497    0.235
                  foot   2438      7110 0.833 0.709 0.772    0.513
  ```
- YOLOv9-Wholebody28 - C - ReLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.800 0.591 0.680    0.512
                  body   2438     13133 0.899 0.766 0.858    0.708
                 adult   2438      9934 0.905 0.696 0.827    0.727
                 child   2438      1062 0.836 0.737 0.820    0.756
                  male   2438      7074 0.871 0.781 0.863    0.755
                female   2438      2983 0.831 0.742 0.824    0.726
  body_with_wheelchair   2438       191 0.899 0.936 0.971    0.867
    body_with_crutches   2438       124 0.957 0.895 0.924    0.909
                  head   2438     11097 0.886 0.891 0.926    0.687
                 front   2438      2113 0.722 0.389 0.502    0.422
           right-front   2438      2096 0.751 0.443 0.564    0.463
            right-side   2438      1265 0.752 0.531 0.633    0.521
            right-back   2438       825 0.725 0.480 0.599    0.492
                  back   2438       633 0.663 0.354 0.473    0.368
             left-back   2438       734 0.756 0.478 0.578    0.474
             left-side   2438      1239 0.763 0.530 0.628    0.527
            left-front   2438      2190 0.747 0.369 0.504    0.415
                  face   2438      5880 0.890 0.851 0.892    0.674
                   eye   2438      5038 0.768 0.380 0.467    0.203
                  nose   2438      4818 0.853 0.566 0.638    0.362
                 mouth   2438      3898 0.768 0.498 0.561    0.281
                   ear   2438      4704 0.786 0.530 0.599    0.341
              shoulder   2438     17126 0.674 0.477 0.510    0.226
                 elbow   2438     10464 0.606 0.381 0.414    0.178
                  hand   2438      7947 0.917 0.621 0.799    0.551
             hand_left   2438      4044 0.888 0.551 0.745    0.519
            hand_right   2438      3903 0.880 0.560 0.736    0.512
                  knee   2438      8756 0.609 0.417 0.454    0.203
                  foot   2438      7110 0.809 0.683 0.742    0.475
  ```
- YOLOv9-Wholebody28 - E - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.855 0.638 0.741    0.584
                  body   2438     13133 0.925 0.807 0.895    0.777
                 adult   2438      9934 0.946 0.719 0.869    0.798
                 child   2438      1062 0.905 0.788 0.870    0.830
                  male   2438      7074 0.914 0.811 0.898    0.819
                female   2438      2983 0.892 0.781 0.869    0.795
  body_with_wheelchair   2438       191 0.897 0.958 0.969    0.892
    body_with_crutches   2438       124 0.970 0.968 0.991    0.974
                  head   2438     11097 0.900 0.931 0.956    0.757
                 front   2438      2113 0.805 0.402 0.558    0.488
           right-front   2438      2096 0.829 0.466 0.628    0.540
            right-side   2438      1265 0.841 0.570 0.692    0.600
            right-back   2438       825 0.824 0.521 0.668    0.574
                  back   2438       633 0.799 0.395 0.546    0.450
             left-back   2438       734 0.812 0.482 0.608    0.524
             left-side   2438      1239 0.846 0.557 0.679    0.596
            left-front   2438      2190 0.820 0.397 0.552    0.476
                  face   2438      5880 0.910 0.873 0.911    0.728
                   eye   2438      5038 0.822 0.449 0.565    0.269
                  nose   2438      4818 0.897 0.614 0.701    0.435
                 mouth   2438      3898 0.837 0.571 0.655    0.350
                   ear   2438      4704 0.846 0.612 0.700    0.423
              shoulder   2438     17126 0.742 0.557 0.604    0.300
                 elbow   2438     10464 0.675 0.476 0.525    0.254
                  hand   2438      7947 0.937 0.667 0.852    0.639
             hand_left   2438      4044 0.920 0.605 0.812    0.609
            hand_right   2438      3903 0.907 0.619 0.802    0.605
                  knee   2438      8756 0.668 0.504 0.550    0.279
                  foot   2438      7110 0.867 0.756 0.811    0.570
  ```
- YOLOv9-Wholebody28 - E - ReLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2438    140381 0.827 0.620 0.717    0.553
                  body   2438     13133 0.913 0.787 0.875    0.743
                 adult   2438      9934 0.925 0.711 0.853    0.768
                 child   2438      1062 0.854 0.757 0.838    0.791
                  male   2438      7074 0.892 0.803 0.888    0.793
                female   2438      2983 0.868 0.761 0.843    0.756
  body_with_wheelchair   2438       191 0.899 0.963 0.978    0.894
    body_with_crutches   2438       124 0.950 0.968 0.985    0.964
                  head   2438     11097 0.892 0.918 0.948    0.731
                 front   2438      2113 0.754 0.393 0.530    0.454
           right-front   2438      2096 0.795 0.452 0.591    0.497
            right-side   2438      1265 0.797 0.539 0.665    0.564
            right-back   2438       825 0.785 0.502 0.632    0.528
                  back   2438       633 0.737 0.363 0.510    0.408
             left-back   2438       734 0.743 0.469 0.576    0.487
             left-side   2438      1239 0.792 0.542 0.655    0.565
            left-front   2438      2190 0.775 0.384 0.528    0.450
                  face   2438      5880 0.900 0.870 0.908    0.710
                   eye   2438      5038 0.806 0.437 0.543    0.250
                  nose   2438      4818 0.889 0.606 0.685    0.415
                 mouth   2438      3898 0.819 0.549 0.626    0.329
                   ear   2438      4704 0.828 0.593 0.674    0.403
              shoulder   2438     17126 0.710 0.527 0.568    0.261
                 elbow   2438     10464 0.643 0.434 0.474    0.213
                  hand   2438      7947 0.928 0.660 0.837    0.602
             hand_left   2438      4044 0.906 0.590 0.791    0.574
            hand_right   2438      3903 0.895 0.603 0.779    0.560
                  knee   2438      8756 0.627 0.458 0.499    0.235
                  foot   2438      7110 0.829 0.731 0.788    0.531
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
  @software{YOLOv9-Wholebody28,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: Body, Adult, Child, Male, Female, Body_with_Wheelchair, Body_with_Crutches, Head, Front, Right_Front, Right_Side, Right_Back, Back, Left_Back, Left_Side, Left_Front, Face, Eye, Nose, Mouth, Ear, Shoulder, Elbow, Hand, Hand_Left, Hand_Right, Knee, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/464_YOLOv9-Wholebody28},
    year={2025},
    month={02},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/464_YOLOv9-Wholebody28/LICENSE)

## 7. Next Challenge
- Wrist, Hip, Ankle

  https://github.com/user-attachments/assets/a15d4bc3-b593-4261-874f-7214dc8fd3c7

- Steps and final goal

  ![image](https://github.com/user-attachments/assets/d5974b9e-018b-4739-99ae-1e5f879c0c3f)
