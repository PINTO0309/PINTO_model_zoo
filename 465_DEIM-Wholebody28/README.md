# 465_DEIM-Wholebody28

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

This model far surpasses the performance of existing CNNs in both inference speed and accuracy. I'm not particularly interested in comparing performance between architectures, so I don't cherry-pick any of the verification results. What is important is a balance between accuracy, speed, the number of output classes, and versatility of output values.

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `Body_with_Wheelchair`, `Body_with_Crutches`, `Head`, `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side`, `Left_Front`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Shoulder`, `Elbow`, `Hand`, `Hand_Left`, `Hand_Right`, `Knee`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

The aim is to estimate head pose direction with minimal computational cost using only an object detection model, with an emphasis on practical aspects. The concept is significantly different from existing full-mesh type head direction estimation models, head direction estimation models with tweaked loss functions, and models that perform precise 360° 6D estimation. Capturing the features of every part of the body on a 2D surface makes it very easy to combine with other feature extraction processes.

Don't be ruled by the curse of mAP.

- Difficulty: Normal

- Difficulty: Normal

  https://www2.nhk.or.jp/archives/movies/?id=D0002160854_00000

  https://github.com/user-attachments/assets/b54d77e8-9ba2-4752-81c5-cdfdc1263101

- Difficulty: Super Hard
  - The depression and elevation angles are quite large.
  - People move quickly. (Intense motion blur)
  - The image quality is quite poor and there is a lot of noise. (Quality taken around 1993)
  - The brightness is quite dark.

  https://www2.nhk.or.jp/archives/movies/?id=D0002080169_00000

  https://github.com/user-attachments/assets/ba45ac9b-ae1b-45be-a41c-de10f0f4ffa6

- Difficulty: Super Ultra Hard (Score threshold 0.35)
  - Heavy Rain.
  - High intensity halation.
  - People move quickly. (Intense motion blur)
  - The image quality is quite poor and there is a lot of noise. (Quality taken around 2003)
  - The brightness is quite dark.

  https://www2.nhk.or.jp/archives/movies/?id=D0002040195_00000

  https://github.com/user-attachments/assets/b9ff6e7a-98d3-42b7-921d-72287b1fe2c1

- Difficulty: Super Hard (Score threshold 0.35)

  https://www.pakutaso.com/20240833234post-51997.html

  ![shikunHY5A3705_TP_V](https://github.com/user-attachments/assets/1e3e57ff-60b2-4799-9d42-86cb6a38836f)

- Other results

  |output<br>`Objects score threshold >= 0.65`<br>`Attributes score threshold >= 0.70`<br>`Keypoints score threshold >= 0.35`<br>`1,250 query`|output<br>`Objects score threshold >= 0.65`<br>`Attributes score threshold >= 0.70`<br>`Keypoints score threshold >= 0.35`<br>`1,250 query`|
  |:-:|:-:|
  |![000000003786](https://github.com/user-attachments/assets/72eeaf52-2d2a-4496-bf37-eeeafa188f4a)|![000000005673](https://github.com/user-attachments/assets/830cf3d4-70db-4d40-875c-ad73c7a168c2)|
  |![000000009420](https://github.com/user-attachments/assets/ac7cc347-97a1-4a88-8fba-7cd56fade0fa)|![000000010082](https://github.com/user-attachments/assets/fa046ee3-df14-4a46-b09d-2acaaf4080b5)|
  |![000000010104](https://github.com/user-attachments/assets/6e7b5cb9-b64a-4c8a-b65f-3b55e04605f6)|![000000015778](https://github.com/user-attachments/assets/ba09ec7d-5ec9-4f73-a9a6-a35ca5547143)|
  |![000000048658](https://github.com/user-attachments/assets/e94af5b6-7505-4acf-8fb2-ac1ae5820c0d)|![000000048893](https://github.com/user-attachments/assets/304dc1fc-c359-468c-9e29-af49faa9c3c1)|
  |![000000061606](https://github.com/user-attachments/assets/fa58a170-c61a-4f6e-b75b-ab9e3f45cd5d)|![000000064824](https://github.com/user-attachments/assets/0d45ed94-0e80-4025-a261-aba6285070e4)|
  |![000000065500](https://github.com/user-attachments/assets/df0c4b38-c47d-4ee2-9dd2-8c6debe1ccc6)|![000000066754](https://github.com/user-attachments/assets/73e5ce6d-0430-4697-86a9-6ad0ff654115)|
  |![000000073470](https://github.com/user-attachments/assets/96fa0355-e2c4-43ef-a1ee-5cb3c448d95b)|![000000094388](https://github.com/user-attachments/assets/a63518fa-a8c4-46cc-9b1f-491a1822dcce)|
  |![frameE_000002](https://github.com/user-attachments/assets/30f39b53-e5ff-4a21-ba69-e8e978691c5e)|![frameE_000071](https://github.com/user-attachments/assets/538c4869-b155-4f05-9fb9-c6e8f71b2a71)|

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

  The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`.

  https://github.com/user-attachments/assets/32f150fe-ebf7-4374-9f0d-f1130badcfc1

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
  - RTX3070 (VRAM: 8GB)
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
  - `score_threshold` is a very rough value set for testing purposes, so feel free to adjust it to your liking. The default threshold is probably too low.
  - There is a lot of information being rendered into the image, so if you want to compare performance with other models it is best to run the demo with `-dnm`, `-dgm`, `-dlr` and `-dhm`.

    ```
    usage:
      demo_rtdetrv2_onnx_wholebody25.py \
      [-h] \
      [-m MODEL] \
      (-v VIDEO | -i IMAGES_DIR) \
      [-ep {cpu,cuda,tensorrt}] \
      [-it] \
      [-ost] \
      [-ast] \
      [-dvw] \
      [-dwk] \
      [-dnm] \
      [-dgm] \
      [-dlr] \
      [-dhm] \
      [-oyt] \
      [-bblw]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for DEIM-Wholebody28.
      -v VIDEO, --video VIDEO
        Video file path or camera index.
      -i IMAGES_DIR, --images_dir IMAGES_DIR
        jpg, png images folder path.
      -ep {cpu,cuda,tensorrt}, \
          --execution_provider {cpu,cuda,tensorrt}
        Execution provider for ONNXRuntime.
      -it {fp16,int8}, --inference_type {fp16,int8}
        Inference type. Default: fp16
      -ost OBJECT_SCORE_THRESHOLD, --object_score_threshold OBJECT_SCORE_THRESHOLD
        Object score threshold. Default: 0.65
      -ast ATTRIBUTE_SCORE_THRESHOLD, --attribute_score_threshold ATTRIBUTE_SCORE_THRESHOLD
        Attribute score threshold. Default: 0.70
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
      -bblw BOUNDING_BOX_LINE_WIDTH, --bounding_box_line_width BOUNDING_BOX_LINE_WIDTH
        Bounding box line width. Default: 2
    ```

- DEIM-Wholebody28 - S - 1,250 query
  ```
  ```
- DEIM-Wholebody28 - X - 1,250 query
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.660
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.832
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.705
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.479
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.835
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.922
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.343
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.718
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.575
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.888
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.956
  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.887
  Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.763  
  ```

- Pre-Process

  To ensure fair benchmark comparisons with YOLOX, `BGR to RGB conversion processing` and `normalization by division by 255.0` are added to the model input section.

  ![image](https://github.com/user-attachments/assets/9098f12d-6b04-497f-947e-02bc77855f51)

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{DEIM-Wholebody28,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 25 classes: Body, Adult, Child, Male, Female, Body_with_Wheelchair, Body_with_Crutches, Head, Front, Right_Front, Right_Side, Right_Back, Back, Left_Back, Left_Side, Left_Front, Face, Eye, Nose, Mouth, Ear, Hand, Hand_Left, Hand_Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/465_DEIM-Wholebody28},
    year={2024},
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

  - DEIM

    https://github.com/ShihuaHuang95/DEIM

    ```bibtex
    @misc{huang2024deim,
          title={DEIM: DETR with Improved Matching for Fast Convergence},
          author={Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen},
          year={2024},
          eprint={2412.04234},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    ```

  - PINTO Custom DEIM (Drastically change the training parameters and optimize the model structure)

    https://github.com/PINTO0309/DEIM

## 6. License
[Apache2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/465_DEIM-Wholebody28/LICENSE)

## 7. Next challenge
- Wrist, Hip, Ankle
- Steps and final goal

  ![image](https://github.com/user-attachments/assets/d5974b9e-018b-4739-99ae-1e5f879c0c3f)

