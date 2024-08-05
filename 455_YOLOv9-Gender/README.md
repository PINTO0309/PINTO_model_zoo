# YOLOv9-Gender

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 3 classes: `Body`, `Male`, `Female`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

This model does not use facial features, but only whole-body features to estimate gender. In other words, gender can be estimated even when the body is turned backwards and the face cannot be seen at all. This model is transfer learning using YOLOv9-Wholebody13 weights.

Don't be ruled by the curse of mAP.

|Sample|Sample|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/9deea5c3-3799-41b3-bfd0-44686756e415)|![image](https://github.com/user-attachments/assets/b461e250-92f2-4982-91dd-9fb775e3da4a)|
|![000000090862](https://github.com/user-attachments/assets/9a7ee3ae-a4d1-4872-9924-577b0c2c231d)|![000000089648](https://github.com/user-attachments/assets/b977937c-5f46-41e4-8a78-e7a711a1bd23)|

The `s` key on the keyboard can be used to enable or disable the gender recognition mode.

https://github.com/user-attachments/assets/cc0ff8a6-5756-47f4-bece-142200a02e00

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

  Halfway compromises are never acceptable.

  ![image](https://github.com/user-attachments/assets/a0a4c053-4004-4cb5-97e5-a35605bde827)

  |Class Name|Class ID|Remarks|
  |:-|-:|:-|
  |Body|0|Detection accuracy is higher than `Male` and `Female` bounding boxes. It is the sum of `Male`, `Female`, and `Unknown` labels.|
  |Male|1|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Female|2|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |(Unknown)|3|It is not included in the classes output from the model because it uses tricks that are ignored during training. I use them as annotation-only labels for fairly small objects or when noise effects make it impossible for me to visually determine. 14,288 labels.|

  ![image](https://github.com/user-attachments/assets/52b0514c-96a1-468a-aa54-af0de916854f)

## 3. Test
  - Python 3.10
  - onnx 1.14.1+
  - onnxruntime-gpu v1.16.1 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.16.1 + CUDA 11.8 + TensorRT 8.5.3 build (RTX3070)](https://zenn.dev/pinto0309/scraps/20afd3c58b30bf))
  - opencv-contrib-python 4.9.0.80+
  - numpy 1.24.3
  - TensorRT 8.5.3-1+cuda11.8

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
      demo_yolov9_onnx_gender.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dgm]

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
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode.
    ```

- YOLOv9-Gender - N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
   Class Images Instances     P     R mAP50 mAP50-95
     all   2386     23317 0.625 0.548 0.579    0.416
    body   2386     13172 0.742 0.653 0.722    0.493
    male   2386      7157 0.646 0.600 0.612    0.462
  female   2386      2988 0.488 0.391 0.402    0.293
  ```
- YOLOv9-Gender - T - Swish/SiLU
  ```
   Class Images Instances     P     R mAP50 mAP50-95
     all   2386     23317 0.747 0.647 0.715    0.566
    body   2386     13172 0.856 0.743 0.827    0.622
    male   2386      7157 0.738 0.685 0.732    0.599
  female   2386      2988 0.647 0.514 0.587    0.476
  ```
- YOLOv9-Gender - S - Swish/SiLU
  ```
  ```
- YOLOv9-Gender - M - Swish/SiLU
  ```
  ```
- YOLOv9-Gender - C - Swish/SiLU
  ```
  ```
- YOLOv9-Gender - E - Swish/SiLU
  ```
   Class Images Instances     P     R mAP50 mAP50-95
     all   2386     23317 0.887 0.785 0.869    0.766
    body   2386     13172 0.919 0.869 0.930    0.787
    male   2386      7157 0.885 0.778 0.863    0.782
  female   2386      2988 0.857 0.709 0.814    0.729
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
    --input_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --output_onnx_file_path yolov9_e_gender_post_0245_1x3x544x960.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Gender,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 3 classes: Body, Male, Female.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/455_YOLOv9-Gender},
    year={2024},
    month={7},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/455_YOLOv9-Gender/LICENSE)
