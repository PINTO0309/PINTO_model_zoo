# 472_DEIMv2-Wholebody34

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 34 classes: `body`, `adult`, `child`, `male`, `female`, `body_with_wheelchair`, `body_with_crutches`, `head`, `front`, `right-front`, `right-side`, `right-back`, `back`, `left-back`, `left-side`, `left-front`, `face`, `eye`, `nose`, `mouth`, `ear`, `collarbone`, `shoulder`, `solar_plexus`, `elbow`, `wrist`, `hand`, `hand_left`, `hand_right`, `abdomen`, `hip_joint`, `knee`, `ankle`, `foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

A notable feature of this model is that it can estimate the shoulder, elbow, and knee joints using only the object detection architecture. That is, I did not use any Pose Estimation architecture, nor did I use human joint keypoint data for training data. Therefore, it is now possible to estimate most of a person's parts, attributes, and keypoints through one-shot inference using a purely traditional simple object detection architecture. By not forcibly combining multiple architectures, inference performance is maximized and training costs are minimized. The difficulty of detecting the elbow is very high.

Don't be ruled by the curse of mAP.

- Image files

  |Image|Image|
  |:-:|:-:|
  |![000000009420](https://github.com/user-attachments/assets/a12b8f9d-0277-4a3c-8f06-faa58cfc06f8)|![000000014428](https://github.com/user-attachments/assets/f62fe90f-4933-4702-a0c3-438ded0790cd)|
  |![000000087933](https://github.com/user-attachments/assets/578af428-0d04-408f-ad59-786d6e21848d)|![000000048658](https://github.com/user-attachments/assets/f650f11e-afd5-4647-8615-f0b6dfbe26ec)|

- USBCam realtime - DEIMv2-X - 1,750 query

  https://github.com/user-attachments/assets/f6643d1d-e4a3-4d90-a673-0a82733294f1

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

  <img width="958" height="850" alt="image" src="https://github.com/user-attachments/assets/cc423513-3136-48fa-a310-3d411d5ee3f4" />


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

## 3. Test
  - Python 3.10+
  - onnx 1.18.1+
  - onnxruntime-gpu v1.22.0
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

  - Demonstration of models
    ```
    usage:
    demo_deimv2_onnx_wholebody34_with_edges.py
    [-h] [-m MODEL] (-v VIDEO | -i IMAGES_DIR) [-ep {cpu,cuda,tensorrt}]
    [-it {fp16,int8}] [-dvw] [-dwk] [-ost OBJECT_SOCRE_THRESHOLD]
    [-ast ATTRIBUTE_SOCRE_THRESHOLD] [-kst KEYPOINT_THRESHOLD]
    [-kdm {dot,box,both}] [-ebm] [-dnm] [-dgm] [-dlr] [-dhm]
    [-drc [DISABLE_RENDER_CLASSIDS ...]] [-efm] [-oyt]
    [-bblw BOUNDING_BOX_LINE_WIDTH]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for DEIMv2.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -i IMAGES_DIR, --images_dir IMAGES_DIR
        jpg, png images folder path.
      -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -it {fp16,int8}, --inference_type {fp16,int8}
        Inference type. Default: fp16
      -dvw, --disable_video_writer
        Disable video writer. Eliminates the file I/O load associated with automatic
        recording to MP4. Devices that use a MicroSD card or similar for main storage
        can speed up overall processing.
      -dwk, --disable_waitKey
        Disable cv2.waitKey(). When you want to process a batch of still images,
        disable key-input wait and process them continuously.
      -ost OBJECT_SOCRE_THRESHOLD, --object_socre_threshold OBJECT_SOCRE_THRESHOLD
        The detection score threshold for object detection. Default: 0.35
      -ast ATTRIBUTE_SOCRE_THRESHOLD, --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
        The attribute score threshold for object detection. Default: 0.70
      -kst KEYPOINT_THRESHOLD, --keypoint_threshold KEYPOINT_THRESHOLD
        The keypoint score threshold for object detection. Default: 0.25
      -kdm {dot,box,both}, --keypoint_drawing_mode {dot,box,both}
        Key Point Drawing Mode. Default: dot
      -ebm, --enable_bone_drawing_mode
        Enable bone drawing mode. (Press B on the keyboard to switch modes)
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
<summary>DEIMv2-Wholebody34 - Atto - 320x320 - 340 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - Femto - 416x416 - 340 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │   27│AP @ .5:.95     │025.35╎AR maxDets   1  │019.05│
  │   27│AP @     .5     │041.90╎AR maxDets  10  │033.47│
  │   27│AP @    .75     │025.52╎AR maxDets 100  │035.01│
  │   27│AP  (small)     │011.05╎AR     (small)  │018.86│
  │   27│AP (medium)     │044.68╎AR    (medium)  │059.68│
  │   27│AP  (large)     │066.36╎AR     (large)  │077.40│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.4206│ 20│ear                      │ 0.1558│
  │  1│adult                    │ 0.4340│ 21│collarbone               │ 0.0650│
  │  2│child                    │ 0.3248│ 22│shoulder                 │ 0.1134│
  │  3│male                     │ 0.4013│ 23│solar_plexus             │ 0.0433│
  │  4│female                   │ 0.3069│ 24│elbow                    │ 0.0789│
  │  5│body_with_wheelchair     │ 0.4803│ 25│wrist                    │ 0.0682│
  │  6│body_with_crutches       │ 0.4859│ 26│hand                     │ 0.2965│
  │  7│head                     │ 0.4403│ 27│hand_left                │ 0.2577│
  │  8│front                    │ 0.3192│ 28│hand_right               │ 0.2538│
  │  9│right-front              │ 0.3115│ 29│abdomen                  │ 0.1077│
  │ 10│right-side               │ 0.3613│ 30│hip_joint                │ 0.0639│
  │ 11│right-back               │ 0.2646│ 31│knee                     │ 0.0931│
  │ 12│back                     │ 0.2162│ 32│ankle                    │ 0.0958│
  │ 13│left-back                │ 0.2917│ 33│foot                     │ 0.2184│
  │ 14│left-side                │ 0.3771│   │                         │       │
  │ 15│left-front               │ 0.3418│   │                         │       │
  │ 16│face                     │ 0.4349│   │                         │       │
  │ 17│eye                      │ 0.1312│   │                         │       │
  │ 18│nose                     │ 0.2124│   │                         │       │
  │ 19│mouth                    │ 0.1530│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - Pico - 640x640 - 340 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │   27│AP @ .5:.95     │036.93╎AR maxDets   1  │023.46│
  │   27│AP @     .5     │056.62╎AR maxDets  10  │041.96│
  │   27│AP @    .75     │038.40╎AR maxDets 100  │043.91│
  │   27│AP  (small)     │019.46╎AR     (small)  │027.80│
  │   27│AP (medium)     │059.35╎AR    (medium)  │068.64│
  │   27│AP  (large)     │078.22╎AR     (large)  │082.95│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.5335│ 20│ear                      │ 0.2548│
  │  1│adult                    │ 0.5730│ 21│collarbone               │ 0.1273│
  │  2│child                    │ 0.5669│ 22│shoulder                 │ 0.1780│
  │  3│male                     │ 0.5698│ 23│solar_plexus             │ 0.0901│
  │  4│female                   │ 0.5425│ 24│elbow                    │ 0.1600│
  │  5│body_with_wheelchair     │ 0.6915│ 25│wrist                    │ 0.1587│
  │  6│body_with_crutches       │ 0.6325│ 26│hand                     │ 0.4446│
  │  7│head                     │ 0.5356│ 27│hand_left                │ 0.4117│
  │  8│front                    │ 0.4032│ 28│hand_right               │ 0.4029│
  │  9│right-front              │ 0.4036│ 29│abdomen                  │ 0.2023│
  │ 10│right-side               │ 0.4766│ 30│hip_joint                │ 0.1411│
  │ 11│right-back               │ 0.4023│ 31│knee                     │ 0.1728│
  │ 12│back                     │ 0.3316│ 32│ankle                    │ 0.1778│
  │ 13│left-back                │ 0.4099│ 33│foot                     │ 0.3259│
  │ 14│left-side                │ 0.4745│   │                         │       │
  │ 15│left-front               │ 0.4314│   │                         │       │
  │ 16│face                     │ 0.5214│   │                         │       │
  │ 17│eye                      │ 0.2357│   │                         │       │
  │ 18│nose                     │ 0.3197│   │                         │       │
  │ 19│mouth                    │ 0.2537│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - N - 640x640 - 680 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │   27│AP @ .5:.95     │039.42╎AR maxDets   1  │024.44│
  │   27│AP @     .5     │058.44╎AR maxDets  10  │044.64│
  │   27│AP @    .75     │040.74╎AR maxDets 100  │048.41│
  │   27│AP  (small)     │020.53╎AR     (small)  │031.69│
  │   27│AP (medium)     │065.44╎AR    (medium)  │075.53│
  │   27│AP  (large)     │085.69╎AR     (large)  │089.89│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.5737│ 20│ear                      │ 0.2649│
  │  1│adult                    │ 0.6062│ 21│collarbone               │ 0.1299│
  │  2│child                    │ 0.5856│ 22│shoulder                 │ 0.1989│
  │  3│male                     │ 0.6050│ 23│solar_plexus             │ 0.1055│
  │  4│female                   │ 0.5495│ 24│elbow                    │ 0.1821│
  │  5│body_with_wheelchair     │ 0.8026│ 25│wrist                    │ 0.1706│
  │  6│body_with_crutches       │ 0.7307│ 26│hand                     │ 0.4862│
  │  7│head                     │ 0.5493│ 27│hand_left                │ 0.4584│
  │  8│front                    │ 0.4213│ 28│hand_right               │ 0.4517│
  │  9│right-front              │ 0.4159│ 29│abdomen                  │ 0.2439│
  │ 10│right-side               │ 0.4862│ 30│hip_joint                │ 0.1800│
  │ 11│right-back               │ 0.3993│ 31│knee                     │ 0.2003│
  │ 12│back                     │ 0.3306│ 32│ankle                    │ 0.1965│
  │ 13│left-back                │ 0.4265│ 33│foot                     │ 0.3695│
  │ 14│left-side                │ 0.4927│   │                         │       │
  │ 15│left-front               │ 0.4330│   │                         │       │
  │ 16│face                     │ 0.5429│   │                         │       │
  │ 17│eye                      │ 0.2351│   │                         │       │
  │ 18│nose                     │ 0.3177│   │                         │       │
  │ 19│mouth                    │ 0.2623│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - S - 640x640 - 1750 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │   27│AP @ .5:.95     │054.27╎AR maxDets   1  │029.00│
  │   27│AP @     .5     │074.93╎AR maxDets  10  │054.87│
  │   27│AP @    .75     │056.55╎AR maxDets 100  │061.57│
  │   27│AP  (small)     │035.08╎AR     (small)  │047.44│
  │   27│AP (medium)     │082.74╎AR    (medium)  │086.38│
  │   27│AP  (large)     │096.02╎AR     (large)  │096.92│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.6777│ 20│ear                      │ 0.4234│
  │  1│adult                    │ 0.7226│ 21│collarbone               │ 0.2604│
  │  2│child                    │ 0.7563│ 22│shoulder                 │ 0.3158│
  │  3│male                     │ 0.7245│ 23│solar_plexus             │ 0.2770│
  │  4│female                   │ 0.7101│ 24│elbow                    │ 0.3182│
  │  5│body_with_wheelchair     │ 0.9078│ 25│wrist                    │ 0.3161│
  │  6│body_with_crutches       │ 0.9146│ 26│hand                     │ 0.6750│
  │  7│head                     │ 0.6560│ 27│hand_left                │ 0.6594│
  │  8│front                    │ 0.5356│ 28│hand_right               │ 0.6512│
  │  9│right-front              │ 0.5458│ 29│abdomen                  │ 0.4282│
  │ 10│right-side               │ 0.6306│ 30│hip_joint                │ 0.3684│
  │ 11│right-back               │ 0.5622│ 31│knee                     │ 0.3390│
  │ 12│back                     │ 0.5139│ 32│ankle                    │ 0.3491│
  │ 13│left-back                │ 0.5925│ 33│foot                     │ 0.5280│
  │ 14│left-side                │ 0.6299│   │                         │       │
  │ 15│left-front               │ 0.5555│   │                         │       │
  │ 16│face                     │ 0.6441│   │                         │       │
  │ 17│eye                      │ 0.3997│   │                         │       │
  │ 18│nose                     │ 0.4583│   │                         │       │
  │ 19│mouth                    │ 0.4040│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 340 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │    5│AP @ .5:.95     │054.65╎AR maxDets   1  │028.90│
  │    5│AP @     .5     │075.99╎AR maxDets  10  │055.43│
  │    5│AP @    .75     │057.05╎AR maxDets 100  │061.76│
  │    5│AP  (small)     │036.13╎AR     (small)  │047.98│
  │    5│AP (medium)     │080.43╎AR    (medium)  │084.84│
  │    5│AP  (large)     │094.43╎AR     (large)  │095.79│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.6888│ 20│ear                      │ 0.4536│
  │  1│adult                    │ 0.7270│ 21│collarbone               │ 0.2497│
  │  2│child                    │ 0.7752│ 22│shoulder                 │ 0.3224│
  │  3│male                     │ 0.7273│ 23│solar_plexus             │ 0.2533│
  │  4│female                   │ 0.7144│ 24│elbow                    │ 0.3242│
  │  5│body_with_wheelchair     │ 0.9288│ 25│wrist                    │ 0.3363│
  │  6│body_with_crutches       │ 0.9458│ 26│hand                     │ 0.6652│
  │  7│head                     │ 0.6582│ 27│hand_left                │ 0.6589│
  │  8│front                    │ 0.5356│ 28│hand_right               │ 0.6461│
  │  9│right-front              │ 0.5528│ 29│abdomen                  │ 0.3966│
  │ 10│right-side               │ 0.6269│ 30│hip_joint                │ 0.3362│
  │ 11│right-back               │ 0.5538│ 31│knee                     │ 0.3463│
  │ 12│back                     │ 0.5096│ 32│ankle                    │ 0.3549│
  │ 13│left-back                │ 0.5854│ 33│foot                     │ 0.5433│
  │ 14│left-side                │ 0.6274│   │                         │       │
  │ 15│left-front               │ 0.5555│   │                         │       │
  │ 16│face                     │ 0.6580│   │                         │       │
  │ 17│eye                      │ 0.4130│   │                         │       │
  │ 18│nose                     │ 0.4814│   │                         │       │
  │ 19│mouth                    │ 0.4284│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 680 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │    5│AP @ .5:.95     │054.90╎AR maxDets   1  │028.93│
  │    5│AP @     .5     │076.52╎AR maxDets  10  │055.50│
  │    5│AP @    .75     │057.23╎AR maxDets 100  │062.07│
  │    5│AP  (small)     │036.67╎AR     (small)  │048.45│
  │    5│AP (medium)     │080.68╎AR    (medium)  │084.93│
  │    5│AP  (large)     │094.48╎AR     (large)  │095.55│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.6894│ 20│ear                      │ 0.4541│
  │  1│adult                    │ 0.7354│ 21│collarbone               │ 0.2480│
  │  2│child                    │ 0.7854│ 22│shoulder                 │ 0.3231│
  │  3│male                     │ 0.7338│ 23│solar_plexus             │ 0.2500│
  │  4│female                   │ 0.7201│ 24│elbow                    │ 0.3307│
  │  5│body_with_wheelchair     │ 0.9228│ 25│wrist                    │ 0.3382│
  │  6│body_with_crutches       │ 0.9354│ 26│hand                     │ 0.6691│
  │  7│head                     │ 0.6559│ 27│hand_left                │ 0.6627│
  │  8│front                    │ 0.5376│ 28│hand_right               │ 0.6506│
  │  9│right-front              │ 0.5590│ 29│abdomen                  │ 0.3963│
  │ 10│right-side               │ 0.6299│ 30│hip_joint                │ 0.3402│
  │ 11│right-back               │ 0.5646│ 31│knee                     │ 0.3493│
  │ 12│back                     │ 0.5184│ 32│ankle                    │ 0.3554│
  │ 13│left-back                │ 0.5813│ 33│foot                     │ 0.5480│
  │ 14│left-side                │ 0.6307│   │                         │       │
  │ 15│left-front               │ 0.5565│   │                         │       │
  │ 16│face                     │ 0.6611│   │                         │       │
  │ 17│eye                      │ 0.4171│   │                         │       │
  │ 18│nose                     │ 0.4832│   │                         │       │
  │ 19│mouth                    │ 0.4314│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 1750 query</summary>

  ```
  ┏━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━┓
  ┃Epoch┃Avg. Precision  ┃     %╇Avg. Recall     ┃     %┃
  ┡━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━┩
  │    6│AP @ .5:.95     │053.83╎AR maxDets   1  │028.56│
  │    6│AP @     .5     │075.67╎AR maxDets  10  │054.63│
  │    6│AP @    .75     │055.85╎AR maxDets 100  │061.57│
  │    6│AP  (small)     │035.33╎AR     (small)  │047.79│
  │    6│AP (medium)     │080.02╎AR    (medium)  │084.59│
  │    6│AP  (large)     │093.83╎AR     (large)  │095.24│
  └─────┴────────────────┴──────┴────────────────┴──────┘
  Per-class mAP:
  ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
  ┃ ID┃Name                     ┃     AP┃ ID┃Name                     ┃     AP┃
  ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
  │  0│body                     │ 0.6864│ 20│ear                      │ 0.4329│
  │  1│adult                    │ 0.7306│ 21│collarbone               │ 0.2410│
  │  2│child                    │ 0.7713│ 22│shoulder                 │ 0.3087│
  │  3│male                     │ 0.7337│ 23│solar_plexus             │ 0.2403│
  │  4│female                   │ 0.7178│ 24│elbow                    │ 0.3128│
  │  5│body_with_wheelchair     │ 0.9276│ 25│wrist                    │ 0.3163│
  │  6│body_with_crutches       │ 0.9487│ 26│hand                     │ 0.6577│
  │  7│head                     │ 0.6470│ 27│hand_left                │ 0.6511│
  │  8│front                    │ 0.5364│ 28│hand_right               │ 0.6389│
  │  9│right-front              │ 0.5478│ 29│abdomen                  │ 0.3855│
  │ 10│right-side               │ 0.6171│ 30│hip_joint                │ 0.3284│
  │ 11│right-back               │ 0.5491│ 31│knee                     │ 0.3328│
  │ 12│back                     │ 0.5182│ 32│ankle                    │ 0.3350│
  │ 13│left-back                │ 0.5799│ 33│foot                     │ 0.5383│
  │ 14│left-side                │ 0.6230│   │                         │       │
  │ 15│left-front               │ 0.5461│   │                         │       │
  │ 16│face                     │ 0.6443│   │                         │       │
  │ 17│eye                      │ 0.3880│   │                         │       │
  │ 18│nose                     │ 0.4631│   │                         │       │
  │ 19│mouth                    │ 0.4053│   │                         │       │
  └───┴─────────────────────────┴───────┴───┴─────────────────────────┴───────┘
  ```

</details>

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
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

  - DEIMv2

    https://github.com/Intellindust-AI-Lab/DEIMv2

    ```bibtex
    @article{huang2025deimv2,
      title={Real-Time Object Detection Meets DINOv3},
      author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
      journal={arXiv},
      year={2025}
    }
    ```

- DEIMv2 custom

  https://github.com/PINTO0309/DEIMv2

## 6. License
[MIT](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/472_DEIMv2-Wholebody34/LICENSE)
