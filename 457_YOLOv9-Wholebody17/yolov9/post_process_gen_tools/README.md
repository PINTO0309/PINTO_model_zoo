# Note
- Post-Process INPUTS
  - `predictions`: `float32 [batches, boxes, 5 + classes]`

    * `5 = [center_x, center_y, width, height, score]`
- Post-Process OUTPUTS
  - `batchno_classid_x1y1x2y2_score`: `float32 [final_boxes_count, 7]`

    * NMS boxes
    * final_boxes_count (N) ≠ batches
    * `7 = [batch_no, classid, x1, y1, x2, y2, score]`

![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9d4fecdf-c90e-4e0a-99a5-9c3e61a4cf41)

# How to generate post-processing ONNX
Simply change the following parameters and run all shells.

```bash
./convert_script.sh
```
Rewrite parameter.
```bash
OPSET=13
BATCHES=1
BOXES=5040
CLASSES=13
```
Run.
```bash
sudo chmod +x ./convert_script.sh
./convert_script.sh
```

# How to change NMS parameters
![image](https://user-images.githubusercontent.com/33194443/178084918-af33bfcc-425f-496e-87fb-1331ef7b2b6e.png)

https://github.com/PINTO0309/simple-onnx-processing-tools

Run the script below to directly rewrite the parameters of the ONNX file.

```bash
### max_output_boxes_per_class
sam4onnx \
--op_name main01_nonmaxsuppression${OPSET} \
--input_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--output_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--input_constants main01_max_output_boxes_per_class int64 [10]

### iou_threshold
sam4onnx \
--op_name main01_nonmaxsuppression${OPSET} \
--input_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--output_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--input_constants main01_iou_threshold float32 [0.05]

### score_threshold
sam4onnx \
--op_name main01_nonmaxsuppression${OPSET} \
--input_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--output_onnx_file_path yolov9_t_wholebody13_post_0245_1x3x544x960.onnx \
--input_constants main01_score_threshold float32 [0.25]
```
