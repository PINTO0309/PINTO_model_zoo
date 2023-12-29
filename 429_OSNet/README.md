# Note

Torchreid is a library for deep-learning person re-identification, written in PyTorch <https://pytorch.org/> and developed for our ICCV'19 project, Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>.

## 1. Citation Repository

  https://github.com/KaiyangZhou/deep-person-reid

## 2. ONNX Export

  https://github.com/PINTO0309/deep-person-reid

## 3. Code snippet for calculating `Cosine similarity` (`COS similarity`) and `Euclidean similarity` (`Euclidean similarity`) from feature vectors

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

  The `Euclidean distance` measures the "straight line distance" between two points. However, when used as a similarity, the distance must be converted to a similarity score in some way. One common method is to use the reciprocal of the distance or some other function based on the distance.
  
  ```python
  import torch
  import torch.nn.functional as F
  
  # Obtain feature vectors from images
  with torch.no_grad():
      f1 = model(image1)  # Feature vector of image1
      f2 = model(image2)  # Feature vector of image2
  
  # Calculation of Euclidean distance
  euclidean_distance = F.pairwise_distance(f1, f2, p=2)
  
  # Conversion from distance to similarity (take the reciprocal)
  # Note: If the distance is very small, the inverse will be very large and needs to be handled appropriately
  similarity = 1.0 / euclidean_distance
  print("\033[1;31m The similarity is {}\033[".format(simlarity[0, 0]))
  ```

## 4. Similarity validation

|Model|30 vs 31⬇️|1 vs 2⏫|1 vs 3⏫|1 vs 4⏫|30 vs 1⬇️|
|:-|-:|-:|-:|-:|-:|
|**1. mlfn**||||||
|9cb5a267|0.521|0.609|0.725|0.740|0.558|
|**2. mobilenetv2**||||||
|1dot0_duke|0.496|0.654|0.852|0.773|0.542|
|1dot0_market|0.402|0.781|0.886|0.882|0.556|
|1dot0_msmt|0.522|0.678|0.624|0.621|0.412|
|1dot4_duke|0.518|0.729|0.853|0.779|0.633|
|1dot4_market|0.409|0.717|0.857|0.839|0.574|
|1dot4_msmt|0.503|0.629|0.652|0.714|0.430|
|1|0.430|0.427|0.428|0.429|0.433|
|**3. osnet**||||||
|ain_d_m_c|0.438|0.610|0.692|0.620|0.437|
|ain_ms_d_c|0.424|0.641|0.645|0.692|0.387|
|ain_ms_d_m|0.436|0.585|0.650|0.670|0.479|
|ain_ms_m_c|0.460|0.547|0.706|0.663|0.393|
|ain_x0_25_imagenet|0.546|0.554|0.703|0.669|0.362|
|ain_x0_5_imagenet|0.602|0.588|0.637|0.669|0.508|
|ain_x0_75_imagenet|0.522|0.643|0.686|0.716|0.529|
|ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0|0.509|0.506|0.685|0.628|0.488|
|ain_x1_0_imagenet|0.504|0.579|0.750|0.720|0.500|
|ain_x1_0_market1501_256x128_amsgrad_ep100_lr0|0.426|0.582|0.825|0.785|0.540|
|ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0|0.444|0.514|0.631|0.517|0.353|
|d_m_c|0.400|0.492|0.668|0.628|0.480|
|ibn_d_m_c|0.376|0.512|0.639|0.626|0.488|
|ibn_ms_d_c|0.440|0.642|0.678|0.633|0.428|
|ibn_ms_d_m|0.464|0.630|0.690|0.686|0.454|
|ibn_ms_m_c|0.439|0.575|0.701|0.616|0.432|
|ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0|0.423|0.507|0.703|0.639|0.425|
|ibn_x1_0_imagenet|0.549|0.536|0.761|0.720|0.495|
|ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0|0.361|0.713|0.759|0.763|0.460|
|ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.329|0.387|0.728|0.403|0.273|
|ms_d_c|**0.389**|**0.531**|**0.685**|**0.650**|**0.457**|
|ms_d_m|0.435|0.542|0.649|0.607|0.489|
|ms_m_c|0.426|0.641|0.746|0.726|0.407|
|x0_25_duke_256x128_amsgrad_ep180_stp80_lr0|0.370|0.535|0.755|0.693|0.500|
|x0_25_imagenet|0.517|0.611|0.766|0.749|0.514|
|x0_25_market_256x128_amsgrad_ep180_stp80_lr0|0.385|0.695|0.835|0.866|0.533|
|x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0|0.352|0.536|0.728|0.563|0.380|
|x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.338|0.453|0.683|0.615|0.329|
|x0_5_duke_256x128_amsgrad_ep180_stp80_lr0|0.314|0.637|0.776|0.744|0.431|
|x0_5_imagenet|0.572|0.585|0.712|0.643|0.567|
|x0_5_market_256x128_amsgrad_ep180_stp80_lr0|0.302|0.741|0.885|0.869|0.442|
|x0_5_msmt17_256x128_amsgrad_ep180_stp80_lr0|0.405|0.621|0.711|0.663|0.402|
|x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.276|0.565|0.639|0.478|0.355|
|x0_75_duke_256x128_amsgrad_ep150_stp60_lr0|0.341|0.644|0.764|0.701|0.517|
|x0_75_imagenet|0.577|0.688|0.756|0.778|0.524|
|x0_75_market_256x128_amsgrad_ep150_stp60_lr0|0.351|0.752|0.843|0.895|0.369|
|x0_75_msmt17_256x128_amsgrad_ep150_stp60_lr0|0.427|0.673|0.667|0.671|0.429|
|x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.320|0.423|0.692|0.492|0.294|
|x1_0_duke_256x128_amsgrad_ep150_stp60_lr0|0.444|0.604|0.716|0.607|0.533|
|x1_0_imagenet|0.589|0.520|0.693|0.644|0.554|
|x1_0_market_256x128_amsgrad_ep150_stp60_lr0|0.349|0.746|0.882|0.801|0.514|
|x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0|0.438|0.526|0.655|0.638|0.438|
|x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|**0.341**|**0.476**|**0.686**|**0.504**|**0.285**|
|**4. resnet50**||||||
|fc512_msmt_xent|0.821|0.835|0.859|0.890|0.808|
|msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0|0.418|0.593|0.810|0.752|0.373|
