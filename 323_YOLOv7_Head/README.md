# Note
https://github.com/PINTO0309/crowdhuman_hollywoodhead_yolo_convert

- YOLOv7_Head - mAP 0.752
  ![test_batch2_pred](https://user-images.githubusercontent.com/33194443/183251257-939d935a-4f54-45a6-95f3-123e07dff848.jpg)

- YOLOv7-tiny_Head - mAP 0.752
  ![test_batch2_pred](https://user-images.githubusercontent.com/33194443/183251276-d96e52fb-4805-4087-bb06-7d3afee4d9e8.jpg)

# Benchmark
- YOLOv7-tiny_Head with Post-Process, ONNX TensorRT, RTX3070
  ```bash
  $ sit4onnx --input_onnx_file_path yolov7_tiny_head_0.752_post_480x640.onnx

  INFO: file: yolov7_tiny_head_0.752_post_480x640.onnx
  INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: input shape: [1, 3, 480, 640] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  13.000011444091797 ms
  INFO: avg elapsed time per pred:  1.3000011444091797 ms
  INFO: output_name.1: score shape: [0, 1] dtype: float32
  INFO: output_name.2: batchno_classid_x1y1x2y2 shape: [0, 6] dtype: int64
  ```

- YOLOv7-tiny_Head with Post-Process Float16, ONNX CUDA, RTX3070
  ```bash
  $ sit4onnx --input_onnx_file_path yolov7_tiny_head_0.752_post_480x640.onnx --onnx_execution_provider cuda

  INFO: file: yolov7_tiny_head_0.752_post_480x640.onnx
  INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: input shape: [1, 3, 480, 640] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  35.124778747558594 ms
  INFO: avg elapsed time per pred:  3.5124778747558594 ms
  INFO: output_name.1: score shape: [0, 1] dtype: float32
  INFO: output_name.2: batchno_classid_x1y1x2y2 shape: [0, 6] dtype: int64
  ```

- YOLOv7-tiny_Head with Post-Process Float32, ONNX CPU, Corei9 Gen.10
  ```bash
  $ sit4onnx --input_onnx_file_path yolov7_tiny_head_0.752_post_480x640.onnx --onnx_execution_provider cpu
  
  INFO: file: yolov7_tiny_head_0.752_post_480x640.onnx
  INFO: providers: ['CPUExecutionProvider']
  INFO: input_name.1: input shape: [1, 3, 480, 640] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  178.92169952392578 ms
  INFO: avg elapsed time per pred:  17.892169952392578 ms
  INFO: output_name.1: score shape: [0, 1] dtype: float32
  INFO: output_name.2: batchno_classid_x1y1x2y2 shape: [0, 6] dtype: int64
  ```

# How to change NMS parameters
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7/post_process_gen_tools#how-to-change-nms-parameters
![image](https://user-images.githubusercontent.com/33194443/183257801-bd25214a-79a9-483c-b03e-fcaa4c229837.png)

https://github.com/PINTO0309/simple-onnx-processing-tools

Run the script below to directly rewrite the parameters of the ONNX file.

e.g.
```bash
$ docker run --rm -it \
-v `pwd`:/workdir \
-w /workdir \
pinto0309/simple-onnx-processing-tools:1.0.41

H=480
W=640

### max_output_boxes_per_class
$ sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--output_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--input_constants main01_max_output_boxes_per_class int64 [100]

### iou_threshold
$ sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--output_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--input_constants main01_iou_threshold float32 [0.6]

### score_threshold
$ sam4onnx \
--op_name main01_nonmaxsuppression11 \
--input_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--output_onnx_file_path yolov7_tiny_head_0.752_post_${H}x${W}.onnx \
--input_constants main01_score_threshold float32 [0.2]
```

# How to generate post-processing ONNX
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7/post_process_gen_tools#how-to-generate-post-processing-onnx
