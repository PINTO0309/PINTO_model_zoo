# Note
- INPUTS
  - `predictions`: `float32 [batches, boxes, classes + 5]`
  
    * 5 = [center_x, center_y, width, height, score]
- OUTPUTS
  - `batchno_classid_x1y1x2y2`: `int64 [final_boxes_count, 6]`

    * NMS boxes
    * final_boxes_count (N) ≠ batches
    * 6 = [batch_no, classid, x1, y1, x2, y2]
  - `score`: `float32 [final_boxes_count, 1]`
  
    * NMS box scores
    * final_boxes_count (N) ≠ batches

![nms_yolov7_5040 onnx](https://user-images.githubusercontent.com/33194443/178084831-eaab28b4-cda8-4528-9e7f-f0b9d0dc7ca5.png)

# How to generate post-processing ONNX
Simply change the following parameters and run all shells.

https://github.com/PINTO0309/PINTO_model_zoo/blob/main/307_YOLOv7/post_process_gen_tools/convert_script.txt
```bash
OPSET=11
BATCHES=1
BOXES=5040
CLASSES=80
```

# How to change NMS parameters
![image](https://user-images.githubusercontent.com/33194443/178084918-af33bfcc-425f-496e-87fb-1331ef7b2b6e.png)

https://github.com/PINTO0309/simple-onnx-processing-tools

Run the script below to directly rewrite the parameters of the ONNX file.
```bash
### Number of output boxes for YOLOv7
BOXES=5040

### max_output_boxes_per_class
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path nms_yolov7_${BOXES}.onnx \
--output_onnx_file_path nms_yolov7_${BOXES}.onnx \
--input_constants main01_max_output_boxes_per_class int64 [10]

### iou_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path nms_yolov7_${BOXES}.onnx \
--output_onnx_file_path nms_yolov7_${BOXES}.onnx \
--input_constants main01_iou_threshold float32 [0.6]

### score_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path nms_yolov7_${BOXES}.onnx \
--output_onnx_file_path nms_yolov7_${BOXES}.onnx \
--input_constants main01_score_threshold float32 [0.7]
```

# How to merge post-processing into a YOLOv7 model
Simply execute the following command.

https://github.com/PINTO0309/simple-onnx-processing-tools
```bash
################################################### YOLOv7 + Post-Process
MODEL=yolov7
BOXES=5040
H=256
W=320

snc4onnx \
--input_onnx_file_paths ${MODEL}_${H}x${W}.onnx nms_yolov7_${BOXES}.onnx \
--srcop_destop output predictions \
--output_onnx_file_path ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
onnxsim ${MODEL}_post_${H}x${W}.onnx ${MODEL}_post_${H}x${W}.onnx
```
