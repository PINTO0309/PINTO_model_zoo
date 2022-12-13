# Note
- Post-Process (Myriad Support) - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/337_FreeYOLO/PINTO_special/convert_script.txt
![image](https://user-images.githubusercontent.com/33194443/207271656-0b7fc7ca-aadb-4d3c-b18c-388bd60c687d.png)

# How to change NMS parameter
## 1. max_output_boxes_per_class
  Value that controls the maximum number of bounding boxes output per class. The closer the value is to 1, the faster the overall post-processing speed.
  ```
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
  or
  ```
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
## 2. iou_threshold
  NMS IOU Thresholds.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants iou_threshold float32 [0.5]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants iou_threshold float32 [0.5]
  ```
## 3. score_threshold
  Threshold of scores to be detected for banding boxes. The closer the value is to 1.0, the faster the overall post-processing speed.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants score_threshold float32 [0.75]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --output_onnx_file_path yolo_free_nano_640x640_post.onnx \
  --op_name post_nms_NonMaxSuppression \
  --input_constants score_threshold float32 [0.75]
  ```
