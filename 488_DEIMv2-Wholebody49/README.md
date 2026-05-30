# 488_DEIMv2-Wholebody49

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Unified multi-task model for detection, pose estimation, and instance segmentation.

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 49 classes. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. The quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

A notable feature of this model is that it can estimate the shoulder, elbow, and knee joints using only the object detection architecture. That is, I did not use any Pose Estimation architecture, nor did I use human joint keypoint data for training data. Therefore, it is now possible to estimate most of a person's parts, attributes, and keypoints through one-shot inference using a purely traditional simple object detection architecture. By not forcibly combining multiple architectures, inference performance is maximized and training costs are minimized.

Since the segmentation mask was trained with a fairly small size of `80x80`, you shouldn't expect too much in terms of mask boundary accuracy. Furthermore, a bug in the training pipeline was discovered just before training was completed, resulting in the final generated weights being trained with insufficient precision tuning. Consequently, the weights corresponding to the optimal mAP values ​​were not saved in the final .pth file. I experimentally trained variants `S` and `N`, but they did not achieve the expected accuracy, so I do not recommend them.

**2026.05.30 I have found that generating .onnx from .pth and performing inference with CUDA or TensorRT using FP16 significantly degrades the quality of the segmentation mask. I will be making adjustments from various angles for a while, and may update the publicly released ONNX several times.**

The main aspects of the true potential of the object detection model that I verified with this model are as follows:

1. The critical data density at which the parameters of the Transformer architecture saturate
2. Can the hidden region be estimated using a bounding box?
3. Classification of complex left-right crossings
4. Possibility of handling high elevation and depression angles
5. Inference tolerance to high Roll angles
6. Resistance to real-world noise such as blur and darkness
7. Estimated performance when only a small part of the body is visible in first-person view
8. Instance isolation performance to prevent estimation results from mixing across instances in high-congestion situations
9. Classification of head orientation in multiple directions without adding a special decoder
10. Gender and age generation estimation based on the entire body, not just the face
11. Classification of individuals using wheelchairs or crutches
12. The effectiveness of learning by integrating three types of methods—object detection, skeleton detection, and instance segmentation—into a single architecture (The mathematical validity has not been evaluated)
13. The minimum output size required for real-time inference with a single ancient GPU
14. Deployability to edge devices
15. Resolution for objects smaller than 1 pixel
16. Estimation accuracy when using a fisheye camera
17. This method estimates skeletal parts without incorporating constraints on distance and angle between skeletal keypoints into the learning pipeline
18. Improving tolerance to scaling variance when a large number of images with intentionally distorted aspect ratios are included in the training data
19. The model becomes unconvergent when a class is added from a completely different domain, entirely outside the context of "people." e.g. "phone"
20. Evaluation of instance isolation performance using a bottom-up approach

Don't be ruled by the curse of mAP.

- Image files

  |Image|Image|
  |:-:|:-:|
  |<img width="480" height="360" alt="000000000241" src="https://github.com/user-attachments/assets/30777b41-86e2-46e9-8268-e85a03059fcb" />|<img width="480" height="360" alt="000000000328" src="https://github.com/user-attachments/assets/3867f24b-8d3f-4198-aa62-2ce040552817" />|
  |<img width="480" height="360" alt="000000000474" src="https://github.com/user-attachments/assets/c39565bc-e743-418c-97dd-f41fb4230c26" />|<img width="480" height="360" alt="000000000693" src="https://github.com/user-attachments/assets/aef05a4f-e270-477d-9cb6-e8154819b8ed" />|
  |<img width="480" height="360" alt="000000000716" src="https://github.com/user-attachments/assets/e117960e-7b92-4c12-89c0-94380735cc66" />|<img width="480" height="360" alt="000000000836" src="https://github.com/user-attachments/assets/bd2367c5-b53f-43a4-9787-6022f51497e3" />|
  |<img width="480" height="360" alt="000000048893" src="https://github.com/user-attachments/assets/a332594e-a72e-44eb-97ef-4b962f7b849f" />|<img width="480" height="360" alt="000000000984" src="https://github.com/user-attachments/assets/5e796342-b3ab-442e-9eb4-a9709d15865f" />|
  |<img width="480" height="360" alt="000000001292" src="https://github.com/user-attachments/assets/dce59efc-9262-40e9-9aff-d0c1d788b28f" />|<img width="480" height="360" alt="000000001958" src="https://github.com/user-attachments/assets/fecd727e-9d89-4e3b-891b-29c4907d9aa8" />|

- Visualize the bounding box of a bone (This might be due to a pipeline bug, but the regularization for head estimation doesn't seem to be working properly. This happens occasionally.)

  Below are some aspects of the potential of object detection architectures that may have been overlooked or intentionally avoided.

  1. Joint position estimation in invisible areas
  2. Segmentation mask for a segmented area (right hip)
  3. Correct left/right classification of hands that are crossed
  4. The baby doll placed on the floor in the back left has not been detected
  5. Detection of the solar plexus, which is covered by both hands

  <img width="480" height="360" alt="000000014428" src="https://github.com/user-attachments/assets/a593963a-4db7-40b6-8501-97e68459fa27" />

- Realtime Demo - ObjectDetection + PoseEstimation + InstanceSegmentation

  The mask is trained at a very small size of `80x80`, so the boundaries appear wavy.

  https://github.com/user-attachments/assets/95902d00-c5c6-4db4-82b8-2c3325853785

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

  Halfway compromises are never acceptable. The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`. I manually labeled 2,001,101 items.

  <img width="964" height="1169" alt="image" src="https://github.com/user-attachments/assets/e85c45cb-9256-4891-bd16-4c990deafece" />

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
  |Hip_Joint_Left|37|Keypoints. Bounding box coordinates are shared with `Hip_Joint`. It is defined as a subclass of `Hip_Joint` as a superclass.|
  |Hip_Joint_Right|38|Keypoints. Bounding box coordinates are shared with `Hip_Joint`. It is defined as a subclass of `Hip_Joint` as a superclass.|
  |Knee|39|Keypoints|
  |Knee_Left|40|Keypoints. Bounding box coordinates are shared with `Knee`. It is defined as a subclass of `Knee` as a superclass.|
  |Knee_Right|41|Keypoints. Bounding box coordinates are shared with `Knee`. It is defined as a subclass of `Knee` as a superclass.|
  |Ankle|42|Keypoints|
  |Ankle_Left|43|Keypoints. Bounding box coordinates are shared with `Ankle`. It is defined as a subclass of `Ankle` as a superclass.|
  |Ankle_Right|44|Keypoints. Bounding box coordinates are shared with `Ankle`. It is defined as a subclass of `Ankle` as a superclass.|
  |Foot|45|Detection accuracy is higher than `Foot_Left` and `Foot_Right` bounding boxes. It is the sum of `Foot_Left`, and `Foot_Right`.|
  |Foot_Left|46|Bounding box coordinates are shared with `Foot`. It is defined as a subclass of `Foot` as a superclass.|
  |Foot_Right|47|Bounding box coordinates are shared with `Foot`. It is defined as a subclass of `Foot` as a superclass.|
  |Bone|48|Bounding box used to support skeleton/bone line rendering between keypoints.|

## 3. Test
  - Python 3.10+
  - onnx 1.18.1+
  - onnxruntime-gpu v1.22.0+
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
    uv run python demo/demo_deimv2_onnx_torch_wholebody49_ins.py \
    -r deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query_masks.onnx \
    -v 0 \
    -o outputs/video \
    -d tensorrt \
    --score_threshold 0.35 \
    --mask_threshold 0.5 \
    --disable_generation_identification_mode \
    --disable_gender_identification_mode \
    --disable_left_and_right_label \
    --disable_headpose_identification_mode \
    --disable_head_distance_measurement \
    --disable_tracking \
    --enable-masks \
    --enable_bone_drawing_mode
    ```

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{DEIMv2-Wholebody49,
    author={Katsuya Hyodo},
    title={Unified multi-task model for detection, pose estimation, and instance segmentation. 49 classes.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49},
    year={2026},
    month={05},
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
[Apache2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/488_DEIMv2-Wholebody49/LICENSE)
