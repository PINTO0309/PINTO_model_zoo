# Note
https://github.com/PINTO0309/crowdhuman_hollywoodhead_yolo_convert

- YOLOv7_Head - mAP 0.752
  ![test_batch2_pred](https://user-images.githubusercontent.com/33194443/183251257-939d935a-4f54-45a6-95f3-123e07dff848.jpg)

- YOLOv7-tiny_Head - mAP 0.752
  ![test_batch2_pred](https://user-images.githubusercontent.com/33194443/183251276-d96e52fb-4805-4087-bb06-7d3afee4d9e8.jpg)

# Benchmark
- YOLOv7-tiny_Head with Post-Process, TensorRT
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
