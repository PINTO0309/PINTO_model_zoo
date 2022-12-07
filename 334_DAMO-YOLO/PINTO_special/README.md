# Note
- Post-Process - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/334_DAMO-YOLO/PINTO_special/convert_script.txt
![image](https://user-images.githubusercontent.com/33194443/206089409-30fbd73e-7b43-44e1-a093-e87628bad2c3.png)

# How to change NMS parameter
## 1. max_output_boxes_per_class
  Value that controls the maximum number of bounding boxes output per class. The closer the value is to 1, the faster the overall post-processing speed.
  ```
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
  or
  ```
  sam4onnx \
  --input_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --output_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants max_output_boxes_per_class int64 [5]
  ```
## 2. iou_threshold
  NMS IOU Thresholds.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants iou_threshold float32 [0.7]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --output_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants iou_threshold float32 [0.7]
  ```
## 3. score_threshold
  Threshold of scores to be detected for banding boxes. The closer the value is to 1.0, the faster the overall post-processing speed.
  ```bash
  sam4onnx \
  --input_onnx_file_path nms_base_component.onnx \
  --output_onnx_file_path nms_base_component.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants score_threshold float32 [0.25]
  ```
  or
  ```bash
  sam4onnx \
  --input_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --output_onnx_file_path damoyolo_tinynasL35_M_640x640.onnx \
  --op_name nonmaxsuppression11 \
  --input_constants score_threshold float32 [0.25]
  ```
