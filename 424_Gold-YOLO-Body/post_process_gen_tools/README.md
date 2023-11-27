# Note
- INPUTS
  - `predictions`: `float32 [batches, boxes, 5 + classes]`

    * 5 = [center_x, center_y, width, height, score]
- OUTPUTS
  - `batchno_classid_x1y1x2y2_score`: `float32 [final_boxes_count, 7]`

    * NMS boxes
    * final_boxes_count (N) ≠ batches
    * 7 = [batch_no, classid, x1, y1, x2, y2, score]

![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9d4fecdf-c90e-4e0a-99a5-9c3e61a4cf41)

# How to generate post-processing ONNX
Simply change the following parameters and run all shells.

https://github.com/PINTO0309/PINTO_model_zoo/blob/main/420_Gold-YOLO-Hand/post_process_gen_tools/convert_script.sh
```bash
OPSET=11
BATCHES=1
BOXES=5040
CLASSES=1
```

```bash
sudo chmod +x ./convert_script.sh
./convert_script.sh
```

# How to change NMS parameters
![image](https://user-images.githubusercontent.com/33194443/178084918-af33bfcc-425f-496e-87fb-1331ef7b2b6e.png)

https://github.com/PINTO0309/simple-onnx-processing-tools

Run the script below to directly rewrite the parameters of the ONNX file.
```bash
### Number of output boxes for Gold-YOLO
BOXES=5040

### max_output_boxes_per_class
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--output_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--input_constants main01_max_output_boxes_per_class int64 [10]

### iou_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--output_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--input_constants main01_iou_threshold float32 [0.05]

### score_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--output_onnx_file_path 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--input_constants main01_score_threshold float32 [0.25]
```

# How to merge post-processing into a Gold-YOLO model
Simply execute the following command.

https://github.com/PINTO0309/simple-onnx-processing-tools

```bash
################################################### Gold-YOLO + Post-Process
MODEL=gold_yolo
BOXES=5040
H=256
W=320

snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

################################################### 1 Batch

MODEL=gold_yolo

BOXES=5040
H=256
W=320
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

BOXES=7560
H=256
W=480
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

BOXES=10080
H=256
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

BOXES=15120
H=384
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

BOXES=18900
H=480
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

BOXES=57960
H=736
W=1280
snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx 30_nms_gold_yolo_m_hand_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx

################################################### N Batch

MODEL=gold_yolo

BOXES=5040
H=256
W=320
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx

BOXES=7560
H=256
W=480
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx

BOXES=10080
H=256
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx

BOXES=15120
H=384
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx

BOXES=18900
H=480
W=640
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx

BOXES=57960
H=736
W=1280
snc4onnx \
--input_onnx_file_paths ${MODEL}_Nx3x${H}x${W}.onnx 31_nms_gold_yolo_m_hand_N_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
onnxsim ${MODEL}_post_Nx3x${H}x${W}.onnx ${MODEL}_post_Nx3x${H}x${W}.onnx
```
