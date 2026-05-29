# 488_DEIMv2-Wholebody49

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Unified multi-task model for detection, pose estimation, and instance segmentation. This model is an experimental model I created to evaluate the quality of annotation data I developed myself, and it is incomplete because the keypoints on the left and right sides of the lower body have not yet been labeled. Furthermore, this model was created to verify how accurately learning progresses when multi-task training is performed on data with completely different label properties.

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 49 classes. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. The quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

A notable feature of this model is that it can estimate the shoulder, elbow, and knee joints using only the object detection architecture. That is, I did not use any Pose Estimation architecture, nor did I use human joint keypoint data for training data. Therefore, it is now possible to estimate most of a person's parts, attributes, and keypoints through one-shot inference using a purely traditional simple object detection architecture. By not forcibly combining multiple architectures, inference performance is maximized and training costs are minimized.

Don't be ruled by the curse of mAP.

- Image files

  |Image|Image|
  |:-:|:-:|
  |||


- Realtime Demo - ObjectDetection + PoseEstimation + InstanceSegmentation

  https://github.com/user-attachments/assets/cec11315-bccc-43b3-9609-26e03fed4f02

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

  Halfway compromises are never acceptable. The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`. I manually labeled 1,257,651 items.

  <img width="960" height="965" alt="image" src="https://github.com/user-attachments/assets/ee49eb1b-d2ca-42b2-a55f-75ee5afc3af2" />

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
  |Shoulder_Left|23|Keypoints. Bounding box coordinates are shared with `Shoulder`. It is defined as a subclass of `Shoulder` as a superclass.|
  |Shoulder_Right|24|Keypoints. Bounding box coordinates are shared with `Shoulder`. It is defined as a subclass of `Shoulder` as a superclass.|
  |Solar_Plexus|25|Keypoints|
  |Elbow|26|Keypoints|
  |Elbow_Left|27|Keypoints. Bounding box coordinates are shared with `Elbow`. It is defined as a subclass of `Elbow` as a superclass.|
  |Elbow_Right|28|Keypoints. Bounding box coordinates are shared with `Elbow`. It is defined as a subclass of `Elbow` as a superclass.|
  |Wrist|29|Keypoints|
  |Wrist_Left|30|Keypoints. Bounding box coordinates are shared with `Wrist`. It is defined as a subclass of `Wrist` as a superclass.|
  |Wrist_Right|31|Keypoints. Bounding box coordinates are shared with `Wrist`. It is defined as a subclass of `Wrist` as a superclass.|
  |Hand|32|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|33|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|34|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Abdomen|35|Keypoints|
  |Hip_Joint|36|Keypoints|
  |Knee|37|Keypoints|
  |Ankle|38|Keypoints|
  |Foot (Feet)|39||

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
    uv run python demo/wholebody40/demo_deimv2_torch_wholebody40_ins.py \
    -r deimv2_dinov3_x_wholebody40_ins_s08_maskhead256x3_center_800query_masks.onnx \
    --video 0 \
    -o outputs/video \
    -d tensorrt \
    --score_threshold 0.40 \
    --mask_threshold 0.50 \
    --disable_generation_identification_mode \
    --disable_gender_identification_mode \
    --disable_headpose_identification_mode \
    --disable_head_distance_measurement \
    --enable-masks \
    --mask_bilateral_d 5 \
    --mask_bilateral_sigma_color 1.0 \
    --mask_bilateral_sigma_space 1.0
    ```

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{DEIMv2-Wholebody40,
    author={Katsuya Hyodo},
    title={Unified multi-task model for detection, pose estimation, and instance segmentation. Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 40 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right_front, right_side, right_back, back, left_back, left_side, left_front, face, eye, nose, mouth, ear, collarbone, shoulder, shoulder_left, shoulder_right, solar_plexus, elbow, elbow_left, elbow_right, wrist, wrist_left, wrist_right, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/485_DEIMv2-Wholebody40},
    year={2026},
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
[Apache2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/485_DEIMv2-Wholebody40/LICENSE)
