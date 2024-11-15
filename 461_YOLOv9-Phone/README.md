# 461_YOLOv9-Phone

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight phone detection models.

https://github.com/user-attachments/assets/91fe579d-f36d-440e-ad86-a3f18e481389

|output<br>`Objects score threshold >= 0.35`|output<br>`Objects score threshold >= 0.35`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/0132effe-8d5b-4cd3-b541-ddd590038f8f)|![image](https://github.com/user-attachments/assets/48c99b0c-aaf9-4a68-b879-ba6d66180e76)|
|![image](https://github.com/user-attachments/assets/6b269bdb-cfd1-4c3a-b8a5-1bdf078c47bb)|![image](https://github.com/user-attachments/assets/5567df2d-2fb1-4931-8074-44da5d9ec25b)|
|![image](https://github.com/user-attachments/assets/d27909bb-4e7f-4a30-8feb-3ec83000c87d)|![image](https://github.com/user-attachments/assets/6d36ea30-1011-42dc-b8e0-a5cd47658e41)|
|![image](https://github.com/user-attachments/assets/d8da8fb0-977d-4da6-8dda-c93900b2dd47)|![image](https://github.com/user-attachments/assets/9c440d3b-1564-4474-8702-9cfbab667a00)|
|![image](https://github.com/user-attachments/assets/2520c41e-b57c-451a-a0ab-f3c88f70d24c)|![image](https://github.com/user-attachments/assets/ce7fa4a7-56ae-435b-bcda-c8e2d45c553d)|

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
  - MS-COCO https://cocodataset.org/#download
  - [CD-COCO: Complex Distorted COCO database for Scene-Context-Aware computer vision](https://github.com/aymanbegh/cd-coco)

  ![image](https://github.com/user-attachments/assets/b59580f9-b589-4d24-9634-4d17d8fd4e79)

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
    usage:
      demo_yolov9_onnx_phone.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-ost] \
      [-ast] \
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

- YOLOv9-Phone - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
  Class Images Instances     P     R mAP50 mAP50-95
    all   2007      2676 0.706 0.517 0.582    0.389
  ```
- YOLOv9-Phone - T - Swish/SiLU
  ```
  Class Images Instances     P     R mAP50 mAP50-95
    all   2007      2676 0.792 0.647 0.711    0.505
  ```
- YOLOv9-Phone - S - Swish/SiLU
  ```
  Class Images Instances     P     R mAP50 mAP50-95
    all   2007      2676 0.885 0.704 0.792    0.590
  ```
- YOLOv9-Phone - E - Swish/SiLU
  ```
  Class Images Instances     P     R mAP50 mAP50-95
    all   2007      2676 0.896 0.732 0.805    0.612
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
    --input_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_phone_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 3. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Phone,
    author={Katsuya Hyodo},
    title={Lightweight phone detection models.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/461_YOLOv9-Phone},
    year={2024},
    month={11},
    doi={10.5281/zenodo.10229410}
  }
  ```

## 4. Cited
  I am very grateful for their excellent work.
  - [MS-COCO](https://cocodataset.org/#download)

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
