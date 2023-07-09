# Note

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/2f75d180-eea9-4da6-988f-ba76cd589f00

```
$ sit4onnx -if face_blendshapes.onnx -oep cpu
INFO: file: face_blendshapes.onnx
INFO: providers: ['CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  5.369663238525391 ms
INFO: avg elapsed time per pred:  0.5369663238525391 ms
INFO: output_name.1: output shape: [52] dtype: float32

$ sit4onnx -if face_blendshapes.onnx -oep cuda
INFO: file: face_blendshapes.onnx
INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  28.800249099731445 ms
INFO: avg elapsed time per pred:  2.8800249099731445 ms
INFO: output_name.1: output shape: [52] dtype: float32

$ sit4onnx -if face_blendshapes.onnx -oep tensorrt
INFO: file: face_blendshapes.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input_points shape: [1, 146, 2] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  3.676176071166992 ms
INFO: avg elapsed time per pred:  0.3676176071166992 ms
INFO: output_name.1: output shape: [52] dtype: float32
```
