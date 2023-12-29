# Note

Torchreid is a library for deep-learning person re-identification, written in PyTorch <https://pytorch.org/> and developed for our ICCV'19 project, Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>.

## 1. Citation Repository

  https://github.com/KaiyangZhou/deep-person-reid

## 2. ONNX Export

  https://github.com/PINTO0309/deep-person-reid

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

||image.1|image.2|
|:-|:-:|:-:|
|30 vs 31⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b2249f44-cd26-49da-8796-25e12f2831fe)|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|
|30 vs 1⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/893ed42c-4a63-4779-97e2-2af9ae57a79f)|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8afb01a8-f7c4-483f-9387-62e59d715693)|
|31 vs 2⬇️|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|
|1 vs 2⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/82854902-c63b-4b24-859d-23661fe65f0c)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|
|1 vs 3⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/49f09597-94c8-4130-aa43-b4f3971ed9a7)|![3](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/79ba35d2-88de-4534-9bf5-c1c64d36c279)|
|1 vs 4⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8fae11e3-1a46-4907-85b4-f9a9d3257e47)|![4](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c32a10d9-bb67-484f-8483-4c7080e70312)|

```bash
python validation.py
```

|Model|30 vs 31⬇️|30 vs 1⬇️|31 vs 2⬇️|1 vs 2⏫|1 vs 3⏫|1 vs 4⏫|
|:-|-:|-:|-:|-:|-:|-:|
|**1. mlfn**|||||||
|9cb5a267|0.521|0.558|0.609|0.609|0.725|0.740|
|**2. mobilenetv2**|||||||
|1dot0_duke|0.496|0.542|0.501|0.654|0.852|0.773|
|1dot0_market|0.402|0.556|0.469|0.781|0.886|0.882|
|1dot0_msmt|0.522|0.412|0.471|0.678|0.624|0.621|
|1dot4_duke|0.518|0.633|0.552|0.729|0.853|0.779|
|1dot4_market|0.409|0.574|0.516|0.717|0.857|0.839|
|1dot4_msmt|0.503|0.430|0.425|0.629|0.651|0.714|
|1|0.430|0.433|0.423|0.427|0.428|0.429|
|**3. osnet**|||||||
|ain_d_m_c|0.438|0.437|0.418|0.610|0.692|0.620|
|ain_ms_d_c|0.425|0.387|0.422|0.641|0.645|0.692|
|ain_ms_d_m|0.436|0.479|0.406|0.585|0.650|0.670|
|ain_ms_m_c|0.460|0.393|0.381|0.547|0.706|0.663|
|ain_x0_25_imagenet|0.546|0.362|0.448|0.554|0.703|0.669|
|ain_x0_5_imagenet|0.602|0.508|0.525|0.589|0.637|0.670|
|ain_x0_75_imagenet|0.522|0.529|0.477|0.643|0.686|0.716|
|ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0|0.509|0.488|0.378|0.506|0.685|0.628|
|ain_x1_0_imagenet|0.504|0.500|0.491|0.579|0.750|0.720|
|ain_x1_0_market1501_256x128_amsgrad_ep100_lr0|0.426|0.540|0.461|0.582|0.825|0.785|
|ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0|0.444|0.353|0.349|0.514|0.631|0.517|
|d_m_c|0.400|0.480|0.446|0.492|0.668|0.628|
|ibn_d_m_c|0.376|0.488|0.432|0.512|0.639|0.626|
|ibn_ms_d_c|0.440|0.428|0.373|0.642|0.678|0.633|
|ibn_ms_d_m|0.464|0.454|0.462|0.630|0.690|0.686|
|ibn_ms_m_c|0.439|0.432|0.467|0.575|0.701|0.616|
|ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0|0.423|0.425|0.440|0.506|0.703|0.639|
|ibn_x1_0_imagenet|0.549|0.495|0.552|0.536|0.761|0.720|
|ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0|0.361|0.460|0.535|0.713|0.759|0.763|
|ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.329|0.273|0.281|0.387|0.728|0.403|
|ms_d_c|**0.389**|**0.457**|**0.407**|**0.531**|**0.685**|**0.650**|
|ms_d_m|0.435|0.489|0.436|0.541|0.649|0.607|
|ms_m_c|0.426|0.407|0.492|0.641|0.746|0.726|
|x0_25_duke_256x128_amsgrad_ep180_stp80_lr0|0.370|0.500|0.430|0.535|0.755|0.693|
|x0_25_imagenet|0.517|0.514|0.634|0.611|0.766|0.749|
|x0_25_market_256x128_amsgrad_ep180_stp80_lr0|0.385|0.533|0.405|0.695|0.835|0.866|
|x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0|0.352|0.380|0.332|0.536|0.728|0.563|
|x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.338|0.329|0.348|0.453|0.683|0.615|
|x0_5_duke_256x128_amsgrad_ep180_stp80_lr0|**0.314**|**0.431**|**0.445**|**0.637**|**0.776**|**0.744**|
|x0_5_imagenet|0.572|0.567|0.562|0.585|0.712|0.643|
|x0_5_market_256x128_amsgrad_ep180_stp80_lr0|0.302|0.442|0.412|0.741|0.885|0.869|
|x0_5_msmt17_256x128_amsgrad_ep180_stp80_lr0|0.405|0.402|0.388|0.621|0.711|0.663|
|x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.276|0.355|0.265|0.565|0.639|0.478|
|x0_75_duke_256x128_amsgrad_ep150_stp60_lr0|0.341|0.517|0.453|0.644|0.764|0.701|
|x0_75_imagenet|0.577|0.524|0.604|0.688|0.756|0.778|
|x0_75_market_256x128_amsgrad_ep150_stp60_lr0|0.351|0.369|0.430|0.752|0.843|0.895|
|x0_75_msmt17_256x128_amsgrad_ep150_stp60_lr0|0.427|0.429|0.393|0.673|0.667|0.671|
|x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.320|0.294|0.312|0.423|0.692|0.492|
|x1_0_duke_256x128_amsgrad_ep150_stp60_lr0|0.444|0.533|0.433|0.605|0.716|0.606|
|x1_0_imagenet|0.589|0.554|0.553|0.520|0.693|0.644|
|x1_0_market_256x128_amsgrad_ep150_stp60_lr0|0.349|0.514|0.506|0.746|0.882|0.801|
|x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0|0.438|0.438|0.447|0.526|0.655|0.638|
|x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|**0.341**|**0.285**|**0.265**|**0.476**|**0.686**|**0.504**|
|**4. resnet50**|||||||
|fc512_msmt_xent|0.821|0.808|0.779|0.835|0.859|0.890|
|msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|**0.418**|**0.373**|**0.330**|**0.593**|**0.810**|**0.752**|
