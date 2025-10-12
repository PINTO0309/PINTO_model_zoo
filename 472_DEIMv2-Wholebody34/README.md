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
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - Pico - 640x640 - 340 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - N - 640x640 - 680 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - S - 640x640 - 1750 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 340 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 680 query</summary>

  ```
  ```

</details>
<details>
<summary>DEIMv2-Wholebody34 - X - 640x640 - 1750 query</summary>

  ```
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
