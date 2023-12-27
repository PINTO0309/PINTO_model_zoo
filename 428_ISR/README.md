# Note

- Citation Repository

  https://github.com/dcp15/ISR_ICCV2023_Oral

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

- test
  ```
  usage: demo_isr_onnx_tfite.py \
  [-h] \
  [-m MODEL] \
  [-i1 IMAGE1] \
  [-i2 IMAGE2] \
  [-ep {cpu,cuda,tensorrt}]
  
  options:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL
                          ONNX/TFLite file path for YOLOX.
    -i1 IMAGE1, --image1 IMAGE1
                          Base image file.
    -i2 IMAGE2, --image2 IMAGE2
                          Target image file.
    -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
                          Execution provider for ONNXRuntime.


  python demo/demo_isr_onnx_tfite.py -m isr_2x3x224x224_11.onnx -i1 1.png -i2 2.png
  The similarity is 0.511 Elapsed time: 113.86ms
  
  python demo/demo_isr_onnx_tfite.py -m isr_2x3x224x224_11.onnx -i1 1.png -i2 3.png
  The similarity is 0.764 Elapsed time: 113.91ms
  
  python demo/demo_isr_onnx_tfite.py -m isr_2x3x224x224_11.onnx -i1 1.png -i2 4.png
  The similarity is 0.725 Elapsed time: 156.75ms
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
