# 463_YOLOv9-Shoulder-Elbow-Knee

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

**⚠️ The accuracy is not high because it is an experimental project. ⚠️**

Lightweight Shoulder/Elbow/Knee detection models. This model is implemented using only YOLOv9, and does not use any of the known 2D skeleton detection architectures, so the computational cost is very low. Instead of having a left and right concept, it has succeeded in minimizing the computational cost of key points. This does not use either the “Backbone” or “Head” of the skeleton detection. Therefore, it is possible to detect joints using only the pre- and post-processing of normal object detection. This model is trained to minimize the detection of humans in mascot costumes, mannequins, and synthetic images. In addition, in this model, full-body tights and competitive ski wear are not considered mascot costumes.

- Realtime inference demo

  https://github.com/user-attachments/assets/c4553ef2-985c-4279-a095-a5941dc3dfe3

- This model is a verification model for the first step in implementing a skeleton detection algorithm using only an object detection architecture. Ultimately, I am aiming to add skeletal detection capabilities to [YOLOv9-Wholebody25](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25).
  |Step.1|Step.2|
  |:-:|:-:|
  |![image](https://github.com/user-attachments/assets/20f6bde0-10f1-474e-86f7-94ea7f89db4f)|![image](https://github.com/user-attachments/assets/7b07fbb8-2ca0-4107-82a6-133ae759f130)|

  |Step.3|Step.4|
  |:-:|:-:|
  |![image](https://github.com/user-attachments/assets/5fca644c-a23a-41b3-b873-c238df51e476)|![image](https://github.com/user-attachments/assets/218b477b-5782-4f7a-ac78-0b4b826d3fb8)|

- Sample of detection results

  At present, it is only trained using experimental data sets, so detection of close-range objects is unstable. This is expected to be resolved in `Wholebody28`. `463_YOLOv9-Shoulder-Elbow-Knee` was created for the sole purpose of concept verification.

  |output<br>`Objects score threshold >= 0.35`|output<br>`Objects score threshold >= 0.35`|
  |:-:|:-:|
  |![image](https://github.com/user-attachments/assets/e7490294-8ee2-4f25-b684-f88c5cd7f7ad)|![image](https://github.com/user-attachments/assets/743422cf-8724-4358-9dab-17d90a613961)|
  |![image](https://github.com/user-attachments/assets/72d0fc23-947c-46d5-8433-2e2643535e15)|![image](https://github.com/user-attachments/assets/5d55b16c-d6d8-44d8-9e60-a7efe167315c)|
  |![image](https://github.com/user-attachments/assets/213daafa-849f-43c3-bf11-7fcb3340e366)|![image](https://github.com/user-attachments/assets/2cf8fb77-a5f2-4bae-8820-e0a6bc531f62)|
  |![image](https://github.com/user-attachments/assets/ebd82e9d-4f0c-46a9-81f8-3b6c7575a925)|![image](https://github.com/user-attachments/assets/291ed615-c441-4526-89b7-09e41ac6cab7)|

- Mode

  |`dot`|`box`|`both`|
  |:-:|:-:|:-:|
  |![000000088214](https://github.com/user-attachments/assets/4380d722-e1b4-4ecf-86bb-637d869e73f3)|![000000088214](https://github.com/user-attachments/assets/9974aa0a-8c33-43a4-9313-8bc8e03bad8d)|![000000088214](https://github.com/user-attachments/assets/d1b2a840-63fc-4e59-912f-b11eed60c385)|

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

    ![image](https://github.com/user-attachments/assets/af9325f4-25fa-4513-8d2a-6d5e2da25ee3)

## 2. Test
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
    usage: demo_yolov9_onnx_sholder_elbow_knee.py
      [-h]
      [-m MODEL]
      (-v VIDEO | -i IMAGES_DIR)
      [-ep {cpu,cuda,tensorrt}]
      [-it {fp16,int8}]
      [-dvw]
      [-dwk]
      [-ost OBJECT_SOCRE_THRESHOLD]
      [-kdm {dot,box,both}]
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
      -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -it {fp16,int8}, --inference_type {fp16,int8}
        Inference type. Default: fp16
      -dvw, --disable_video_writer
        Disable video writer. Eliminates the file I/O load associated with automatic recording to MP4.
        Devices that use a MicroSD card or similar for main storage can speed up overall processing.
      -dwk, --disable_waitKey
        Disable cv2.waitKey(). When you want to process a batch of still images,
        disable key-input wait and process them continuously.
      -ost OBJECT_SOCRE_THRESHOLD, --object_socre_threshold OBJECT_SOCRE_THRESHOLD
        The detection score threshold for object detection.
        Default: 0.35
      -kdm {dot,box,both}, --keypoint_drawing_mode {dot,box,both}
        Key Point Drawing Mode. (Press K on the keyboard to switch modes)
        Default: dot
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
    ```

- YOLOv9-Shoulder-Elbow-Knee - N - Swish/SiLU
  - First Step
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      34445      0.436      0.275       0.27     0.0973
    shoulder       1777      16405      0.458      0.339      0.319      0.106
       elbow       1777      10012      0.445      0.222      0.231     0.0842
        knee       1777       8028      0.406      0.263       0.26      0.102
    ```
  - Fine-tuning
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      33796      0.467      0.284      0.289      0.106
    shoulder       1777      16307      0.500      0.342      0.336      0.117
       elbow       1777       9753      0.468      0.227      0.247     0.0903
        knee       1777       7736      0.433      0.282      0.284      0.111
    ```
- YOLOv9-Shoulder-Elbow-Knee - T - Swish/SiLU
  - First Step
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      34445      0.603      0.359      0.400      0.158
    shoulder       1777      16405      0.631      0.410      0.442      0.164
       elbow       1777      10012      0.590      0.313      0.357      0.141
        knee       1777       8028      0.588      0.355      0.400      0.167
    ```
  - Fine-tuning
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      33796      0.641      0.376      0.426      0.176
    shoulder       1777      16307      0.666      0.421      0.463      0.182
       elbow       1777       9753      0.629      0.328      0.379      0.155
        knee       1777       7736      0.628      0.379      0.435      0.190
    ```
- YOLOv9-Shoulder-Elbow-Knee - S - Swish/SiLU
  - First Step
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      34445      0.694      0.434      0.493      0.214
    shoulder       1777      16405      0.730      0.480      0.538      0.226
       elbow       1777      10012      0.671      0.387      0.445      0.191
        knee       1777       8028      0.680      0.433      0.495      0.226
    ```
  - Fine-tuning
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      33796      0.742      0.458      0.532      0.251
    shoulder       1777      16307      0.763      0.492      0.564      0.259
       elbow       1777       9753      0.730      0.415      0.490      0.227
        knee       1777       7736      0.733      0.468      0.544      0.268
    ```
- YOLOv9-Shoulder-Elbow-Knee - C - Swish/SiLU
  - First Step
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      34445      0.766      0.491      0.567      0.273
    shoulder       1777      16405      0.790      0.531      0.608      0.286
       elbow       1777      10012      0.752      0.451      0.526      0.251
        knee       1777       8028      0.756      0.492      0.568      0.283
    ```
  - Fine-tuning
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      33796      0.821      0.536      0.630      0.344
    shoulder       1777      16307      0.832      0.561      0.656      0.351
       elbow       1777       9753      0.813      0.490      0.583      0.313
        knee       1777       7736      0.819      0.556      0.650      0.368
    ```
- YOLOv9-Shoulder-Elbow-Knee - E - Swish/SiLU
  - First Step
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      34445      0.880      0.648      0.747      0.478
    shoulder       1777      16405      0.882      0.658      0.759      0.485
       elbow       1777      10012      0.877      0.619      0.719      0.449
        knee       1777       8028      0.881      0.668      0.763      0.501
    ```
  - Fine-tuning
    ```
       Class     Images  Instances          P          R      mAP50   mAP50-95
         all       1777      33796      0.892      0.660      0.762      0.501
    shoulder       1777      16307      0.902      0.661      0.770      0.502
       elbow       1777       9753      0.882      0.621      0.723      0.465
        knee       1777       7736      0.893      0.699      0.792      0.536
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
    --input_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_sek_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 3. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Shoulder-Elbow-Knee,
    author={Katsuya Hyodo},
    title={Lightweight Shoulder/Elbow/Knee detection models.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/463_YOLOv9-Shoulder-Elbow-Knee},
    year={2025},
    month={1},
    doi={10.5281/zenodo.10229410}
  }
  ```

## 4. Cited
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

## 5. License
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/459_YOLOv9-Wholebody25/LICENSE)
