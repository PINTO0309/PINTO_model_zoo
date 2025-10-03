# 471_YOLO-Wholebody34

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

⚠ The dataset I created became so large that YOLO no longer provided sufficient accuracy for my needs. Therefore, I recommend using the DEIMv2 model, which will be released soon. ⚠

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 34 classes: `body`, `adult`, `child`, `male`, `female`, `body_with_wheelchair`, `body_with_crutches`, `head`, `front`, `right-front`, `right-side`, `right-back`, `back`, `left-back`, `left-side`, `left-front`, `face`, `eye`, `nose`, `mouth`, `ear`, `collarbone`, `shoulder`, `solar_plexus`, `elbow`, `wrist`, `hand`, `hand_left`, `hand_right`, `abdomen`, `hip_joint`, `knee`, `ankle`, `foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

A notable feature of this model is that it can estimate the shoulder, elbow, and knee joints using only the object detection architecture. That is, I did not use any Pose Estimation architecture, nor did I use human joint keypoint data for training data. Therefore, it is now possible to estimate most of a person's parts, attributes, and keypoints through one-shot inference using a purely traditional simple object detection architecture. By not forcibly combining multiple architectures, inference performance is maximized and training costs are minimized. The difficulty of detecting the elbow is very high.

This model is MIT license YOLO.

Don't be ruled by the curse of mAP.

|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|output<br>`Objects score threshold >= 0.35`<br>`Attributes score threshold >= 0.75`<br>`Keypoints score threshold >= 0.25`|
|:-:|:-:|
|||
|||

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
  |Collarbone|21|Keypoints|
  |Shoulder|22|Keypoints|
  |Solar_Plexus|23|Keypoints|
  |Elbow|24|Keypoints|
  |Wrist|25|Keypoints|
  |Hand|26|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|27|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|28|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Abdomen|29|Keypoints|
  |Hip_Joint|30|Keypoints|
  |Knee|31|Keypoints|
  |Ankle|32|Keypoints|
  |Foot (Feet)|33||

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
    ```
    usage:
      demo_yolov9_onnx_wholebody34.py \
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
<summary>YOLO-Wholebody34 - N - Swish/SiLU (PINTO original implementation, 2.4 MB)</summary>

  ```
  ┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ Epoch ┃ Avg. Precision ┃     % ┃ Avg. Recall    ┃     % ┃
  ┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │    0  │ AP @ .5:.95    │ 30.07 │ AR maxDets   1 │ 27.92 │
  │    0  │ AP @     .5    │ 45.67 │ AR maxDets  10 │ 45.61 │
  │    0  │ AP @    .75    │ 31.32 │ AR maxDets 100 │ 48.85 │
  │    0  │ AP  (small)    │  8.23 │ AR     (small) │ 25.05 │
  │    0  │ AP (medium)    │ 34.14 │ AR    (medium) │ 61.46 │
  │    0  │ AP  (large)    │ 60.41 │ AR     (large) │ 76.80 │
  └───────┴────────────────┴───────┴────────────────┴───────┘
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.6061│ 20│ear                      │ 0.2016│
  │  1│adult                    │ 0.5967│ 21│collarbone               │ 0.0516│
  │  2│child                    │ 0.4263│ 22│shoulder                 │ 0.1921│
  │  3│male                     │ 0.5925│ 23│solar_plexus             │ 0.0364│
  │  4│female                   │ 0.4296│ 24│elbow                    │ 0.1071│
  │  5│body_with_wheelchair     │ 0.6396│ 25│wrist                    │ 0.0531│
  │  6│body_with_crutches       │ 0.6060│ 26│hand                     │ 0.2683│
  │  7│head                     │ 0.6195│ 27│hand_left                │ 0.2065│
  │  8│front                    │ 0.4475│ 28│hand_right               │ 0.2143│
  │  9│right-front              │ 0.4695│ 29│abdomen                  │ 0.0865│
  │ 10│right-side               │ 0.4215│ 30│hip_joint                │ 0.0479│
  │ 11│right-back               │ 0.2672│ 31│knee                     │ 0.1152│
  │ 12│back                     │ 0.1702│ 32│ankle                    │ 0.0832│
  │ 13│left-back                │ 0.3182│ 33│foot                     │ 0.2486│
  │ 14│left-side                │ 0.4152│   │                         │       │
  │ 15│left-front               │ 0.5008│   │                         │       │
  │ 16│face                     │ 0.4962│   │                         │       │
  │ 17│eye                      │ 0.1017│   │                         │       │
  │ 18│nose                     │ 0.2141│   │                         │       │
  │ 19│mouth                    │ 0.1415│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>YOLO-Wholebody34 - T - Swish/SiLU</summary>

  ```
  ┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ Epoch ┃ Avg. Precision ┃     % ┃ Avg. Recall    ┃     % ┃
  ┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │    0  │ AP @ .5:.95    │ 40.82 │ AR maxDets   1 │ 32.82 │
  │    0  │ AP @     .5    │ 57.82 │ AR maxDets  10 │ 53.66 │
  │    0  │ AP @    .75    │ 42.54 │ AR maxDets 100 │ 56.79 │
  │    0  │ AP  (small)    │ 13.07 │ AR     (small) │ 34.92 │
  │    0  │ AP (medium)    │ 51.20 │ AR    (medium) │ 72.20 │
  │    0  │ AP  (large)    │ 75.95 │ AR     (large) │ 83.99 │
  └───────┴────────────────┴───────┴────────────────┴───────┘
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.7427│ 20│ear                      │ 0.2500│
  │  1│adult                    │ 0.7391│ 21│collarbone               │ 0.0946│
  │  2│child                    │ 0.6566│ 22│shoulder                 │ 0.2870│
  │  3│male                     │ 0.7676│ 23│solar_plexus             │ 0.0864│
  │  4│female                   │ 0.7171│ 24│elbow                    │ 0.2120│
  │  5│body_with_wheelchair     │ 0.8434│ 25│wrist                    │ 0.1041│
  │  6│body_with_crutches       │ 0.7017│ 26│hand                     │ 0.4096│
  │  7│head                     │ 0.7009│ 27│hand_left                │ 0.3534│
  │  8│front                    │ 0.5711│ 28│hand_right               │ 0.3501│
  │  9│right-front              │ 0.5714│ 29│abdomen                  │ 0.1793│
  │ 10│right-side               │ 0.5352│ 30│hip_joint                │ 0.1331│
  │ 11│right-back               │ 0.4800│ 31│knee                     │ 0.2331│
  │ 12│back                     │ 0.2886│ 32│ankle                    │ 0.1527│
  │ 13│left-back                │ 0.4166│ 33│foot                     │ 0.3618│
  │ 14│left-side                │ 0.5270│   │                         │       │
  │ 15│left-front               │ 0.5390│   │                         │       │
  │ 16│face                     │ 0.5688│   │                         │       │
  │ 17│eye                      │ 0.1455│   │                         │       │
  │ 18│nose                     │ 0.2733│   │                         │       │
  │ 19│mouth                    │ 0.1977│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>YOLO-Wholebody34 - S - Swish/SiLU</summary>

  ```
  ┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ Epoch ┃ Avg. Precision ┃     % ┃ Avg. Recall    ┃     % ┃
  ┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │    0  │ AP @ .5:.95    │ 54.28 │ AR maxDets   1 │ 38.62 │
  │    0  │ AP @     .5    │ 71.72 │ AR maxDets  10 │ 62.99 │
  │    0  │ AP @    .75    │ 56.77 │ AR maxDets 100 │ 65.15 │
  │    0  │ AP  (small)    │ 21.99 │ AR     (small) │ 44.20 │
  │    0  │ AP (medium)    │ 72.86 │ AR    (medium) │ 83.60 │
  │    0  │ AP  (large)    │ 91.68 │ AR     (large) │ 94.10 │
  └───────┴────────────────┴───────┴────────────────┴───────┘
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.8502│ 20│ear                      │ 0.3280│
  │  1│adult                    │ 0.8530│ 21│collarbone               │ 0.1900│
  │  2│child                    │ 0.8508│ 22│shoulder                 │ 0.4626│
  │  3│male                     │ 0.8938│ 23│solar_plexus             │ 0.2465│
  │  4│female                   │ 0.8870│ 24│elbow                    │ 0.3927│
  │  5│body_with_wheelchair     │ 0.9406│ 25│wrist                    │ 0.1939│
  │  6│body_with_crutches       │ 0.8093│ 26│hand                     │ 0.5606│
  │  7│head                     │ 0.7995│ 27│hand_left                │ 0.5245│
  │  8│front                    │ 0.6984│ 28│hand_right               │ 0.5194│
  │  9│right-front              │ 0.7036│ 29│abdomen                  │ 0.3889│
  │ 10│right-side               │ 0.6891│ 30│hip_joint                │ 0.3219│
  │ 11│right-back               │ 0.6553│ 31│knee                     │ 0.4293│
  │ 12│back                     │ 0.4896│ 32│ankle                    │ 0.2583│
  │ 13│left-back                │ 0.5997│ 33│foot                     │ 0.4999│
  │ 14│left-side                │ 0.6987│   │                         │       │
  │ 15│left-front               │ 0.6927│   │                         │       │
  │ 16│face                     │ 0.6599│   │                         │       │
  │ 17│eye                      │ 0.1969│   │                         │       │
  │ 18│nose                     │ 0.3367│   │                         │       │
  │ 19│mouth                    │ 0.2686│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>YOLO-Wholebody34 - C - Swish/SiLU</summary>

  ```
  ┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ Epoch ┃ Avg. Precision ┃     % ┃ Avg. Recall    ┃     % ┃
  ┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │    0  │ AP @ .5:.95    │ 62.85 │ AR maxDets   1 │ 42.31 │
  │    0  │ AP @     .5    │ 80.51 │ AR maxDets  10 │ 69.96 │
  │    0  │ AP @    .75    │ 65.83 │ AR maxDets 100 │ 71.95 │
  │    0  │ AP  (small)    │ 31.16 │ AR     (small) │ 52.25 │
  │    0  │ AP (medium)    │ 82.23 │ AR    (medium) │ 88.75 │
  │    0  │ AP  (large)    │ 95.05 │ AR     (large) │ 96.60 │
  └───────┴────────────────┴───────┴────────────────┴───────┘
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.8774│ 20│ear                      │ 0.4118│
  │  1│adult                    │ 0.8673│ 21│collarbone               │ 0.3012│
  │  2│child                    │ 0.8937│ 22│shoulder                 │ 0.5711│
  │  3│male                     │ 0.9191│ 23│solar_plexus             │ 0.3910│
  │  4│female                   │ 0.9154│ 24│elbow                    │ 0.5227│
  │  5│body_with_wheelchair     │ 0.9456│ 25│wrist                    │ 0.2945│
  │  6│body_with_crutches       │ 0.8615│ 26│hand                     │ 0.6483│
  │  7│head                     │ 0.8555│ 27│hand_left                │ 0.6188│
  │  8│front                    │ 0.7765│ 28│hand_right               │ 0.6271│
  │  9│right-front              │ 0.7958│ 29│abdomen                  │ 0.5108│
  │ 10│right-side               │ 0.7802│ 30│hip_joint                │ 0.4690│
  │ 11│right-back               │ 0.6757│ 31│knee                     │ 0.5639│
  │ 12│back                     │ 0.5973│ 32│ankle                    │ 0.3604│
  │ 13│left-back                │ 0.7088│ 33│foot                     │ 0.6151│
  │ 14│left-side                │ 0.7686│   │                         │       │
  │ 15│left-front               │ 0.8022│   │                         │       │
  │ 16│face                     │ 0.7360│   │                         │       │
  │ 17│eye                      │ 0.2631│   │                         │       │
  │ 18│nose                     │ 0.4274│   │                         │       │
  │ 19│mouth                    │ 0.3505│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>YOLO-Wholebody34 - E - Swish/SiLU</summary>

  ```
  ┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ Epoch ┃ Avg. Precision ┃     % ┃ Avg. Recall    ┃     % ┃
  ┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │    0  │ AP @ .5:.95    │ 65.85 │ AR maxDets   1 │ 43.51 │
  │    0  │ AP @     .5    │ 83.33 │ AR maxDets  10 │ 72.76 │
  │    0  │ AP @    .75    │ 68.87 │ AR maxDets 100 │ 74.58 │
  │    0  │ AP  (small)    │ 35.57 │ AR     (small) │ 57.24 │
  │    0  │ AP (medium)    │ 84.22 │ AR    (medium) │ 89.91 │
  │    0  │ AP  (large)    │ 94.53 │ AR     (large) │ 96.22 │
  └───────┴────────────────┴───────┴────────────────┴───────┘
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.8940│ 20│ear                      │ 0.4699│
  │  1│adult                    │ 0.8926│ 21│collarbone               │ 0.3352│
  │  2│child                    │ 0.8984│ 22│shoulder                 │ 0.5975│
  │  3│male                     │ 0.9265│ 23│solar_plexus             │ 0.4141│
  │  4│female                   │ 0.9191│ 24│elbow                    │ 0.5543│
  │  5│body_with_wheelchair     │ 0.9341│ 25│wrist                    │ 0.3398│
  │  6│body_with_crutches       │ 0.9299│ 26│hand                     │ 0.6840│
  │  7│head                     │ 0.8625│ 27│hand_left                │ 0.6648│
  │  8│front                    │ 0.7918│ 28│hand_right               │ 0.6624│
  │  9│right-front              │ 0.8145│ 29│abdomen                  │ 0.5434│
  │ 10│right-side               │ 0.8095│ 30│hip_joint                │ 0.5001│
  │ 11│right-back               │ 0.7181│ 31│knee                     │ 0.5873│
  │ 12│back                     │ 0.6145│ 32│ankle                    │ 0.3937│
  │ 13│left-back                │ 0.7376│ 33│foot                     │ 0.6473│
  │ 14│left-side                │ 0.7854│   │                         │       │
  │ 15│left-front               │ 0.8133│   │                         │       │
  │ 16│face                     │ 0.7731│   │                         │       │
  │ 17│eye                      │ 0.3078│   │                         │       │
  │ 18│nose                     │ 0.4842│   │                         │       │
  │ 19│mouth                    │ 0.3984│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
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
    --input_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody34_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLO-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/471_YOLO-Wholebody34},
    year={2025},
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

- YOLO
  - Many bug fixes

    https://github.com/PINTO0309/YOLO

  - Original

    https://github.com/MultimediaTechLab/YOLO

## 6. License
[MIT](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/471_YOLO-Wholebody34/LICENSE)
