# Note

- Docker

  https://github.com/PINTO0309/ISR_ICCV2023_Oral

- For CPU/CUDA

  ```
  opset=18
  ```

- For TensorRT

  ```
  opset=11
  ```

- Benchmark

  ```
  ************************************************** opset=18
  sit4onnx -if isr_2x3x224x224_18.onnx -oep cpu
  
  INFO: file: isr_2x3x224x224.onnx
  INFO: providers: ['CPUExecutionProvider']
  INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  1153.2487869262695 ms
  INFO: avg elapsed time per pred:  115.32487869262695 ms
  INFO: output_name.1: output shape: [1, 1] dtype: float32
  
  sit4onnx -if isr_2x3x224x224_18.onnx -oep cuda
  
  INFO: file: isr_2x3x224x224_18.onnx
  INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  168.84374618530273 ms
  INFO: avg elapsed time per pred:  16.884374618530273 ms
  INFO: output_name.1: output shape: [1, 1] dtype: float32
  
  ************************************************** opset=11
  sit4onnx -if isr_2x3x224x224_11.onnx -oep cpu
  
  INFO: file: isr_2x3x224x224_11.onnx
  INFO: providers: ['CPUExecutionProvider']
  INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  1564.5437240600586 ms
  INFO: avg elapsed time per pred:  156.45437240600586 ms
  INFO: output_name.1: output shape: [1, 1] dtype: float32
  
  sit4onnx -if isr_2x3x224x224_11.onnx -oep cuda
  
  INFO: file: isr_2x3x224x224_11.onnx
  INFO: providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  168.98679733276367 ms
  INFO: avg elapsed time per pred:  16.898679733276367 ms
  INFO: output_name.1: output shape: [1, 1] dtype: float32
  
  sit4onnx -if isr_2x3x224x224_11.onnx -oep tensorrt
  
  INFO: file: isr_2x3x224x224_11.onnx
  INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: input shape: [2, 3, 224, 224] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  49.81803894042969 ms
  INFO: avg elapsed time per pred:  4.981803894042969 ms
  INFO: output_name.1: output shape: [1, 1] dtype: float32
  ```
