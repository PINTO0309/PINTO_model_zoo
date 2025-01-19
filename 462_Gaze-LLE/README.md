# 462_Gaze-LLE

Gaze-LLE provides a streamlined gaze architecture that learns only a lightweight gaze decoder on top of a frozen, pretrained visual encoder (DINOv2). Gaze-LLE learns 1-2 orders of magnitude fewer parameters than prior works and doesn't require any extra input modalities like depth and pose!

---

The reason the processing speed of the demo video appears to be around 30ms is because the heatmap rendering process using Pillow is slow. If you stop the heatmap rendering process by pressing the `A` key on the keyboard, the processing speed will increase to around 17ms. Of the 17ms, 10ms is the inference time for YOLOv9-E. If it is important to increase processing speed, it would be better to use [YOLOv9-N or YOLOv9-T](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25).

- Single person test - `gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx` + ONNX-TensorRT

  https://github.com/user-attachments/assets/b8d45d91-55b4-41fe-b177-ab3497026967

- Multi person test - `gazelle_dinov2_vitb14_inout_1x3x448x448_1xNx4.onnx` + ONNX-TensorRT

  https://github.com/user-attachments/assets/91b30d6f-36dd-4278-95b1-471603e36f7c

- Gaze estimation test when facing backwards

  https://github.com/user-attachments/assets/12c5b44b-328c-4d32-b17c-182ddac564f3

- Switch Heatmap mode (`A` key)

  https://github.com/user-attachments/assets/e87bcd2f-5c21-43c0-8ac6-15d1efb711cc

## 1. Test
  - Python 3.10
  - onnx 1.16.1+
  - onnxruntime-gpu v1.18.1 (TensorRT Execution Provider Enabled Binary. See: [onnxruntime-gpu v1.18.1 + CUDA 12.5 + TensorRT 10.2.0 build (RTX3070)](https://zenn.dev/pinto0309/scraps/801db283883c38)
  - opencv-contrib-python 4.10.0.84+
  - numpy 1.24.3
  - TensorRT 10.2.0.19-1+cuda12.5
  - Pillow or pillow-simd

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
    usage: demo_yolov9_onnx_gazelle.py
    [-h]
    [-om OBJECT_DETECTION_MODEL]
    [-gm GAZELLE_MODEL]
    (-v VIDEO | -i IMAGES_DIR)
    [-ep {cpu,cuda,tensorrt}]
    [-it {fp16,int8}]
    [-dvw]
    [-dwk]
    [-ost OBJECT_SOCRE_THRESHOLD]
    [-ast ATTRIBUTE_SOCRE_THRESHOLD]
    [-cst CENTROID_SOCRE_THRESHOLD]
    [-dnm]
    [-dgm]
    [-dlr]
    [-dhm]
    [-dah]
    [-drc [DISABLE_RENDER_CLASSIDS ...]]
    [-oyt]
    [-bblw BOUNDING_BOX_LINE_WIDTH]

    options:
      -h, --help
        show this help message and exit
      -om OBJECT_DETECTION_MODEL, --object_detection_model OBJECT_DETECTION_MODEL
        ONNX/TFLite file path for YOLOv9.
      -gm GAZELLE_MODEL, --gazelle_model GAZELLE_MODEL
        ONNX/TFLite file path for Gaze-LLE.
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
        The detection score threshold for object detection. Default: 0.35
      -ast ATTRIBUTE_SOCRE_THRESHOLD, --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
        The attribute score threshold for object detection. Default: 0.70
      -cst CENTROID_SOCRE_THRESHOLD, --centroid_socre_threshold CENTROID_SOCRE_THRESHOLD
        The heatmap centroid score threshold. Default: 0.30
      -dnm, --disable_generation_identification_mode
        Disable generation identification mode. (Press N on the keyboard to switch modes)
      -dgm, --disable_gender_identification_mode
        Disable gender identification mode. (Press G on the keyboard to switch modes)
      -dlr, --disable_left_and_right_hand_identification_mode
        Disable left and right hand identification mode. (Press H on the keyboard to switch modes)
      -dhm, --disable_headpose_identification_mode
        Disable HeadPose identification mode. (Press P on the keyboard to switch modes)
      -dah, --disable_attention_heatmap_mode
        Disable Attention Heatmap mode. (Press A on the keyboard to switch modes)
      -drc [DISABLE_RENDER_CLASSIDS ...], --disable_render_classids [DISABLE_RENDER_CLASSIDS ...]
        Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19
      -oyt, --output_yolo_format_text
        Output YOLO format texts and images.
      -bblw BOUNDING_BOX_LINE_WIDTH, --bounding_box_line_width BOUNDING_BOX_LINE_WIDTH
        Bounding box line width. Default: 2
    ```

## 2. Cited
  I am very grateful for their excellent work.
  - Gaze-LLE

    https://github.com/fkryan/gazelle

    ```bibtex
    @article{ryan2024gazelle,
      author       = {Ryan, Fiona and Bati, Ajay and Lee, Sangmin and Bolya, Daniel and Hoffman, Judy and Rehg, James M},
      title        = {Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders},
      journal      = {arXiv preprint arXiv:2412.09586},
      year         = {2024},
    }
    ```

## 3. License
[MIT](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/462_Gaze-LLE/LICENSE)

## 4. Ref
- ONNX custom: https://github.com/PINTO0309/gazelle
- Head Detection model 1: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25
- Head Detection model 2: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/460_RT-DETRv2-Wholebody25
