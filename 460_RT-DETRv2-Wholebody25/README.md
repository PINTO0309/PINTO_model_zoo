# 460_RT-DETRv2-Wholebody25

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10229410.svg)](https://doi.org/10.5281/zenodo.10229410)

This model far surpasses the performance of existing CNNs in both inference speed and accuracy.

Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 25 classes: `Body`, `Adult`, `Child`, `Male`, `Female`, `Body_with_Wheelchair`, `Body_with_Crutches`, `Head`, `Front`, `Right_Front`, `Right_Side`, `Right_Back`, `Back`, `Left_Back`, `Left_Side`, `Left_Front`, `Face`, `Eye`, `Nose`, `Mouth`, `Ear`, `Hand`, `Hand_Left`, `Hand_Right`, `Foot`. Even the classification problem is being attempted to be solved by object detection. There is no need to perform any complex affine transformations or other processing for pre-processing and post-processing of input images. In addition, the resistance to Motion Blur, Gaussian noise, contrast noise, backlighting, and halation is quite strong because it was trained only on images with added photometric noise for all images in the MS-COCO subset of the image set. In addition, about half of the image set was annotated by me with the aspect ratio of the original image substantially destroyed. I manually annotated all images in the dataset by myself. The model is intended to use real-world video for inference and has enhanced resistance to all kinds of noise. Probably stronger than any known model. However, the quality of the known data set and my data set are so different that an accurate comparison of accuracy is not possible.

The aim is to estimate head pose direction with minimal computational cost using only an object detection model, with an emphasis on practical aspects. The concept is significantly different from existing full-mesh type head direction estimation models, head direction estimation models with tweaked loss functions, and models that perform precise 360° 6D estimation. Capturing the features of every part of the body on a 2D surface makes it very easy to combine with other feature extraction processes. In experimental trials, the model was trained to only estimate eight Yaw directions, but I plan to add the ability to estimate five Pitch directions in the future.

Don't be ruled by the curse of mAP.

- Difficulty: Normal

  https://github.com/user-attachments/assets/646ab997-f901-4626-88fe-d274a12c9fda

- Difficulty: Super Hard https://www2.nhk.or.jp/archives/movies/?id=D0002080169_00000

  https://github.com/user-attachments/assets/bb19455d-8c3f-4bfa-abe8-143f16b93388

|output<br>`Objects score threshold >= 0.65`<br>`Attributes score threshold >= 0.70`<br>`1,250 query`|output<br>`Objects score threshold >= 0.65`<br>`Attributes score threshold >= 0.70`<br>`1,250 query`|
|:-:|:-:|
|![image](https://github.com/user-attachments/assets/2b310a9f-1203-4db4-9dc8-2129532e3f0d)|![image](https://github.com/user-attachments/assets/c99fb457-a813-4792-b773-84787298a359)|
|![image](https://github.com/user-attachments/assets/fe6df76e-ce43-49c9-af58-4340c4b9502e)|![image](https://github.com/user-attachments/assets/faf65954-3d4b-4d4c-93c1-9b2573a9858a)|
|![image](https://github.com/user-attachments/assets/e2cbd298-6072-4e28-bd3f-b150a704d4af)|![image](https://github.com/user-attachments/assets/711b73f1-2863-4298-b646-2ad2d527b327)|
|![image](https://github.com/user-attachments/assets/d6e66287-cb5b-4806-8bd0-3781750ada3b)|![image](https://github.com/user-attachments/assets/e1c7f9d1-8752-4fa0-8219-48f222846fc3)|
|![image](https://github.com/user-attachments/assets/0063bdd0-1317-40e5-b44b-dd01468436c2)|![image](https://github.com/user-attachments/assets/c9ea9f7c-e0db-4884-9734-dc6b79db5d05)|
|![image](https://github.com/user-attachments/assets/70c57a45-a8ce-47dc-9bdd-6cca10d1b16a)|![image](https://github.com/user-attachments/assets/19aefce8-61d7-4294-bf3d-865529cae228)|
|![image](https://github.com/user-attachments/assets/258805da-6578-4b05-b3d9-67850a027a03)|![image](https://github.com/user-attachments/assets/5114a254-f410-4db7-a61c-2391b8ccfbfb)|

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

  Halfway compromises are never acceptable. I added `2,611` annotations to the following `480x360` image. The trick to annotation is to not miss a single object and not compromise on a single pixel. The ultimate methodology is to `try your best`.

  ![image](https://github.com/user-attachments/assets/ca0b0b44-4280-49aa-b257-fca8429b3337)

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
  |Hand|21|Detection accuracy is higher than `Hand_Left` and `Hand_Right` bounding boxes. It is the sum of `Hand_Left`, and `Hand_Right`.|
  |Hand_Left|22|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Hand_Right|23|Bounding box coordinates are shared with `Hand`. It is defined as a subclass of `Hand` as a superclass.|
  |Foot (Feet)|24||

  ![image](https://github.com/user-attachments/assets/49f9cbf3-3a9c-4666-84ae-d86148c34866)

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
  - `score_threshold` is a very rough value set for testing purposes, so feel free to adjust it to your liking. The default threshold is probably too low.

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
      [-oyt]

    options:
      -h, --help
        show this help message and exit
      -m MODEL, --model MODEL
        ONNX/TFLite file path for RT-DETRv2-Wholebody25.
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
    ```

- RT-DETRv2-Wholebody25 - S (rtdetrv2_r18vd_120e_wholebody25) - 1,250 query
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.602
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.802
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.653
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.432
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.731
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.867
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.615
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.694
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.553
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.820
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.924
  ```
- RT-DETRv2-Wholebody25 - X (rtdetrv2_r101vd_6x_wholebody25) - 1,250 query
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.841
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.700
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.498
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.899
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.346
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.647
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.727
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.598
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.847
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.938
  ```

- Pre-Process

  To ensure fair benchmark comparisons with YOLOX, `BGR to RGB conversion processing` and `normalization by division by 255.0` are added to the model input section.

  ![image](https://github.com/user-attachments/assets/9098f12d-6b04-497f-947e-02bc77855f51)

## 4. Citiation
  If this work has contributed in any way to your research or business, I would be happy to be cited in your literature.
  ```bibtex
  @software{RT-DETRv2-Wholebody25,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 25 classes: Body, Adult, Child, Male, Female, Body_with_Wheelchair, Body_with_Crutches, Head, Front, Right_Front, Right_Side, Right_Back, Back, Left_Back, Left_Side, Left_Front, Face, Eye, Nose, Mouth, Ear, Hand, Hand_Left, Hand_Right, Foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/460_RT-DETRv2-Wholebody25},
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

  - RT-DETRv2

    https://github.com/lyuwenyu/RT-DETR

    ```bibtex
    @misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
          title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
          author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
          year={2024},
          eprint={2407.17140},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2407.17140},
    }
    ```

  - PINTO Custom RT-DETRv2 (Drastically change the training parameters and optimize the model structure)

    https://github.com/PINTO0309/RT-DETR

## 6. License
[Apache2.0](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/460_RT-DETRv2-Wholebody25/LICENSE)
