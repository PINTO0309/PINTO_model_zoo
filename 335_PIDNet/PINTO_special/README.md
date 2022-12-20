# Note

Specifying a Float32 value for the reduction ratio in `fused_argmax_scale_ratio` improves the inference performance of the model according to the reduction ratio. Inference with low-spec devices such as CPUs can improve inference speed by up to 15% to 20%.

See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency

![image](https://user-images.githubusercontent.com/33194443/206860685-5ae497c9-37e4-4820-9ee9-a5a38301cd61.png)

```
inputs:
  input: float32 [N,C,H,W]
  fused_argmax_scale_ratio: float32 scalar_value
```

![image](https://user-images.githubusercontent.com/33194443/206836584-b0e7814d-1e4b-4ee2-b79d-8e4676a9fa4e.png)
