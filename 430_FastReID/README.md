# Note

FastReID is a research platform that implements state-of-the-art re-identification algorithms.

## 1. Citation Repository

  https://github.com/JDAI-CV/fast-reid

  https://github.com/NirAharon/BoT-SORT

## 2. ONNX Export

  https://github.com/PINTO0309/SMILEtrack

## 3. Code snippet for calculating `Cosine similarity` (`COS similarity`) from feature vectors

  `Cosine similarity` is calculated by dividing the inner product of two vectors by the product of their norms. However, since the vectors here are already normalized, simply computing the inner product results in the cosine similarity.

  ```python
  import torch
  import torch.nn.functional as F

  # Obtain feature vectors from images
  with torch.no_grad():
      f1 = model(image1)  # Feature vector of image1
      f2 = model(image2)  # Feature vector of image2
      # Normalize and convert each vector to the unit norm (length is 1)
      A_feat = F.normalize(f1, dim=1).cpu()
      B_feat = F.normalize(f2, dim=1).cpu()
  simlarity = A_feat.matmul(B_feat.transpose(1, 0)) # inner product of feature vectors
  print("\033[1;31m The similarity is {}\033[".format(simlarity[0, 0]))
  ```

## 4. Similarity validation

|Comparison<br>Patterns|image.1|image.2|Comparison<br>Patterns|image.1|image.2|
|:-|:-:|:-:|:-|:-:|:-:|
|30 vs 31⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b2249f44-cd26-49da-8796-25e12f2831fe)|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|1 vs 2⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/82854902-c63b-4b24-859d-23661fe65f0c)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|
|30 vs 1⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/893ed42c-4a63-4779-97e2-2af9ae57a79f)|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8afb01a8-f7c4-483f-9387-62e59d715693)|1 vs 3⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/49f09597-94c8-4130-aa43-b4f3971ed9a7)|![3](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/79ba35d2-88de-4534-9bf5-c1c64d36c279)|
|31 vs 2⬇️|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|1 vs 4⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8fae11e3-1a46-4907-85b4-f9a9d3257e47)|![4](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c32a10d9-bb67-484f-8483-4c7080e70312)|

```bash
python validation.py
```

  |Model|30<br>vs<br>31<br>⬇️|30<br>vs<br>1<br>⬇️|31<br>vs<br>2<br>⬇️|1<br>vs<br>2<br>⏫|1<br>vs<br>3<br>⏫|1<br>vs<br>4<br>⏫|
  |:-|-:|-:|-:|-:|-:|-:|
  |mot17_sbs_S50_NMx3x256x128_post|0.148|0.046|0.219|0.359|0.611|0.543|
  |mot17_sbs_S50_NMx3x288x128_post|0.154|0.036|0.223|0.375|0.643|0.562|
  |mot17_sbs_S50_NMx3x320x128_post|0.093|0.002|0.180|0.386|0.635|0.631|
  |mot17_sbs_S50_NMx3x352x128_post|0.057|0.000|0.153|0.366|0.642|0.649|
  |mot17_sbs_S50_NMx3x384x128_post|0.044|0.000|0.139|0.359|0.629|0.686|
  |mot20_sbs_S50_NMx3x256x128_post|0.406|0.318|0.309|0.538|0.727|0.778|
  |mot20_sbs_S50_NMx3x288x128_post|0.393|0.288|0.324|0.544|0.724|0.770|
  |mot20_sbs_S50_NMx3x320x128_post|0.372|0.253|0.293|0.543|0.701|0.775|
  |mot20_sbs_S50_NMx3x352x128_post|0.351|0.243|0.301|0.578|0.695|0.756|
  |mot20_sbs_S50_NMx3x384x128_post|0.325|0.226|0.289|0.559|0.698|0.757|
  |**OSNet**|||||||
  |osnet_x1_0_msmt17_combineall_256x128_amsgrad_NMx3x256x128|0.341|0.285|0.265|0.476|0.686|0.504|
  |resnet50_msmt17_combineall_256x128_amsgrad_NMx3x256x128|0.418|0.373|0.329|0.593|0.810|0.752|

## 5. BoT-SORT Implementation by onnxruntime + TensorRT only

https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT

https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/0862d659-511d-4be1-8258-f090b39cc51f
