# YOLOv9-Wholebody17

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 15 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `BodyWithWheelchair`, `BodyWithCrutches`, `Head`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand-Left`, `Hand-Right`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

This model does not use facial features, but only whole-body features to estimate generation/gender. In other words, generation/gender can be estimated even when the body is turned backwards and the face cannot be seen at all. This model is transfer learning using YOLOv9-Wholebody15 weights.

Don't be ruled by the curse of mAP.

|input|output `Score threshold >= 0.35`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/d7e04402-fcdc-4dce-a486-4beb5a0c1f01)|![image](https://github.com/user-attachments/assets/9050020f-5139-46d8-b007-b322739d4354)|
|![image](https://github.com/user-attachments/assets/05c75645-792d-4ea9-9733-d0fa5fea5a8e)|![image](https://github.com/user-attachments/assets/0a6379d4-9b75-46a4-92c8-2393b78f4dea)|
|![image](https://github.com/user-attachments/assets/3bad818d-2b92-4a73-8c5a-57bd477d1ac4)|![image](https://github.com/user-attachments/assets/e6461bd9-3362-417d-bba0-58bdeef048ae)|

https://github.com/user-attachments/assets/5f054e24-2b16-454e-9a64-b137974c442f

- The `g` key on the keyboard can be used to enable or disable the gender recognition mode.
- The `h` key on the keyboard can be used to enable or disable the hand recognition mode.

  https://github.com/user-attachments/assets/479c899f-82e0-4a7c-9ca1-670ad5ed78a7

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

  ![image](https://github.com/user-attachments/assets/bf80b1dc-0656-427c-a793-b3d6bfe417e5)

  |Class Name|Class ID|Remarks|
  |:-|-:|:-|
  |Body|0|Detection accuracy is higher than `Adult`, `Child`, `Male` and `Female` bounding boxes. It is the sum of `Adult`, `Child`, `Male`, and `Female`.|
  |Adult|1|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Child|2|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Male|3|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Female|4|Bounding box coordinates are shared with `Body`. It is defined as a subclass of `Body` as a superclass.|
  |Body_with_Wheelchair|5||
  |Body_with_Crutches|6||
  |Head|7||
  |Face|8||
  |Eye|9||
  |Nose|10||
  |Mouth|11||
  |Ear|12||
  |Hand|13|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|14|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|15|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Foot (Feet)|16||

  ![image](https://github.com/user-attachments/assets/bfe210d7-37d1-4ecc-a569-aaa82b821da0)

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
      demo_yolov9_onnx_wholebody17.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-dvw] \
      [-dwk] \
      [-dlr] \
      [-dgm] \
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
      -dlr, --disable_left_and_right_hand_identification_mode
        Disable left and right hand identification mode.
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode.
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
    ```

- YOLOv9-Wholebody17- N - Swish/SiLU (PINTO original implementation, 2.4 MB)
  ```
  WIP
  ```
- YOLOv9-Wholebody17 - T - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2423    103243  0.74 0.552 0.614    0.434
                  body   2423     15581 0.857 0.553 0.658    0.508
                 adult   2423     11474 0.787 0.572 0.638    0.525
                 child   2423      1212 0.526 0.495 0.517    0.436
                  male   2423      8693 0.712 0.616 0.650    0.528
                female   2423      3293 0.571 0.517 0.542    0.451
  body_with_wheelchair   2423       169 0.848 0.894 0.938    0.831
    body_with_crutches   2423       112 0.750 0.857 0.888    0.833
                  head   2423     12981 0.823 0.752 0.800    0.552
                  face   2423      6511 0.856 0.693 0.755    0.517
                   eye   2423      5436 0.675 0.284 0.339    0.132
                  nose   2423      5119 0.728 0.422 0.479    0.245
                 mouth   2423      4162 0.689 0.367 0.410    0.179
                   ear   2423      5149 0.733 0.409 0.459    0.235
                  hand   2423      8124 0.875 0.541 0.675    0.402
             hand_left   2423      4139 0.736 0.425 0.541    0.333
            hand_right   2423      3985 0.721 0.436 0.539    0.332
                  foot   2423      7103 0.690 0.554 0.604    0.340
  ```
- YOLOv9-Wholebody17 - T - ReLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2423    103243 0.718 0.533 0.590    0.409
                  body   2423     15581 0.852 0.535 0.640    0.486
                 adult   2423     11474 0.772 0.561 0.616    0.497
                 child   2423      1212 0.481 0.434 0.453    0.380
                  male   2423      8693 0.700 0.596 0.624    0.498
                female   2423      3293 0.525 0.494 0.504    0.413
  body_with_wheelchair   2423       169 0.830 0.898 0.938    0.800
    body_with_crutches   2423       112 0.673 0.865 0.878    0.799
                  head   2423     12981 0.806 0.739 0.787    0.537
                  face   2423      6511 0.849 0.689 0.751    0.504
                   eye   2423      5436 0.672 0.269 0.320    0.122
                  nose   2423      5119 0.723 0.404 0.458    0.227
                 mouth   2423      4162 0.654 0.353 0.390    0.167
                   ear   2423      5149 0.725 0.386 0.434    0.220
                  hand   2423      8124 0.853 0.515 0.645    0.375
             hand_left   2423      4139 0.710 0.395 0.506    0.305
            hand_right   2423      3985 0.703 0.407 0.509    0.307
                  foot   2423      7103 0.671 0.528 0.574    0.318
  ```
- YOLOv9-Wholebody17 - S - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2423    103243 0.819 0.623 0.698    0.521
                  body   2423     15581 0.893 0.604 0.708    0.581
                 adult   2423     11474 0.857 0.611 0.707    0.616
                 child   2423      1212 0.704 0.621 0.676    0.604
                  male   2423      8693 0.790 0.674 0.731    0.624
                female   2423      3293 0.710 0.617 0.670    0.585
  body_with_wheelchair   2423       169 0.921 0.923 0.963    0.891
    body_with_crutches   2423       112 0.878 0.884 0.922    0.891
                  head   2423     12981 0.854 0.803 0.850    0.613
                  face   2423      6511 0.897 0.708 0.779    0.584
                   eye   2423      5436 0.739 0.328 0.414    0.174
                  nose   2423      5119 0.818 0.490 0.561    0.309
                 mouth   2423      4162 0.779 0.438 0.506    0.239
                   ear   2423      5149 0.766 0.478 0.536    0.294
                  hand   2423      8124 0.907 0.654 0.774    0.509
             hand_left   2423      4139 0.826 0.551 0.677    0.451
            hand_right   2423      3985 0.825 0.560 0.680    0.455
                  foot   2423      7103 0.758 0.651 0.706    0.433
  ```
- YOLOv9-Wholebody17 - C - Swish/SiLU
  ```
  WIP
  ```
- YOLOv9-Wholebody17 - E - Swish/SiLU
  ```
                 Class Images Instances     P     R mAP50 mAP50-95
                   all   2423    103243 0.885 0.697 0.790    0.634
                  body   2423     15581 0.923 0.667 0.775    0.684
                 adult   2423     11474 0.921 0.643 0.778    0.724
                 child   2423      1212 0.825 0.721 0.782    0.739
                  male   2423      8693 0.879 0.718 0.800    0.731
                female   2423      3293 0.860 0.688 0.769    0.708
  body_with_wheelchair   2423       169 0.926 0.964 0.982    0.927
    body_with_crutches   2423       112 0.955 0.940 0.982    0.969
                  head   2423     12981 0.888 0.853 0.899    0.710
                  face   2423      6511 0.915 0.802 0.868    0.704
                   eye   2423      5436 0.806 0.441 0.549    0.262
                  nose   2423      5119 0.884 0.595 0.682    0.427
                 mouth   2423      4162 0.831 0.554 0.643    0.351
                   ear   2423      5149 0.847 0.591 0.666    0.412
                  hand   2423      8124 0.928 0.668 0.847    0.641
             hand_left   2423      4139 0.909 0.611 0.788    0.599
            hand_right   2423      3985 0.904 0.626 0.789    0.602
                  foot   2423      7103 0.842 0.769 0.826    0.582
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
    --input_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --input_constants main01_max_output_boxes_per_class int64 [100]

    ### iou_threshold
    ### Example of changing the allowable area of occlusion to 20%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --input_constants main01_iou_threshold float32 [0.20]

    ### score_threshold
    ### Example of changing the bounding box score threshold to 15%.
    sam4onnx \
    --op_name main01_nonmaxsuppression13 \
    --input_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --output_onnx_file_path yolov9_e_wholebody17_post_0100_1x3x480x640.onnx \
    --input_constants main01_score_threshold float32 [0.15]
    ```
  - Post-processing structure

    PyTorch alone cannot generate this post-processing. For operational flexibility, `EfficientNMS` is not used.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/3c5b34aa-113e-4126-b60d-8532ac91c5b2)

- INT8 quantization ([YOLOv9-QAT](https://zenn.dev/link/comments/1c5e0044f34e45))

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{YOLOv9-Wholebody17,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 17 classes: Body, Adult, Child, Male, Female, BodyWithWheelchair, BodyWithCrutches, Head, Face, Eye, Nose, Mouth, Ear, Hand, Hand-Left, Hand-Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/457_YOLOv9-Wholebody17},
    year={2024},
    month={8},
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
[GPLv3](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/457_YOLOv9-Wholebody17/LICENSE)
