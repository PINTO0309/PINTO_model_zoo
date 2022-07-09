# Note
![nms_yolov7_5040 onnx](https://user-images.githubusercontent.com/33194443/178084831-eaab28b4-cda8-4528-9e7f-f0b9d0dc7ca5.png)

# How to change NMS parameters
![image](https://user-images.githubusercontent.com/33194443/178084918-af33bfcc-425f-496e-87fb-1331ef7b2b6e.png)

https://github.com/PINTO0309/simple-onnx-processing-tools
```bash
### max_output_boxes_per_class
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants main01_max_output_boxes_per_class int64 [10]

### iou_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants main01_iou_threshold float32 [0.6]

### score_threshold
sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants main01_score_threshold float32 [0.7]
```
