# Note

Torchreid is a library for deep-learning person re-identification, written in PyTorch <https://pytorch.org/> and developed for our ICCV'19 project, Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>.

- Citation Repository

  https://github.com/KaiyangZhou/deep-person-reid

- ONNX Export

  https://github.com/PINTO0309/deep-person-reid

- Code snippet for calculating `Cosine similarity` (`COS similarity`) and `Euclidean similarity` (`Euclidean similarity`) from feature vectors

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
