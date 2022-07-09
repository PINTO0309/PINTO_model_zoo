# Note
- INPUTS
  - `predictions`: `float32 [batches, boxes, classes + 5]`
- OUTPUTS
  - `batchno_classid_x1y1x2y2`: `int64 [final_boxes_count, 6]`

    * NMS boxes
    * final_boxes_count (N) ≠ batches
    * 6 = [batch_no, classid, x1, y1, x2, y2]
  - `score`: `float32 [final_boxes_count, 1]`
  
    * final_boxes_count (N) ≠ batches, NMS boxes

![nms_yolov7_5040 onnx](https://user-images.githubusercontent.com/33194443/178084831-eaab28b4-cda8-4528-9e7f-f0b9d0dc7ca5.png)

# How to change NMS parameters
![image](https://user-images.githubusercontent.com/33194443/178084918-af33bfcc-425f-496e-87fb-1331ef7b2b6e.png)

https://github.com/PINTO0309/simple-onnx-processing-tools
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
