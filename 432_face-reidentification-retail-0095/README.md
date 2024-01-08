# face-reidentification-retail-0095

## Use Case and High-Level Description

This is a lightweight network for the face re-identification scenario. It is based on MobileNet V2 backbone, which consists of 3x3 inverted residual blocks with squeeze-excitation attention modules. Instead of the ReLU6 activations used in the original MobileNet V2, this network uses PReLU ones. After the backbone, the network applies global depthwise pooling and then uses 1x1 convolution to create the final embedding vector. The model produces feature vectors which should be close in cosine distance for similar faces and far for different faces.

## Example

![face-reidentification-retail-0095](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/4efdc70a-503b-43e3-b18b-b918981ef781)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| LFW accuracy                    | 0.9947                                    |
| Face location requirements      | Tight aligned crop                        |
| GFlops                          | 0.588                                     |
| MParams                         | 1.107                                     |
| Source framework                | PyTorch\*                                  |

LFW metric is the accuracy in the pairwise re-identification test. See the full [benchmark description](http://vis-www.cs.umass.edu/lfw/) for details.

The model achieves the best results if an input face is frontally oriented and aligned. Face image is aligned if five keypoints (left eye, right eye, tip of nose, left lip corner, right lip corner) are located in the following points in normalized coordinates [0,1]x[0,1]:

```
[(0.31556875000000000, 0.4615741071428571),
 (0.68262291666666670, 0.4615741071428571),
 (0.50026249999999990, 0.6405053571428571),
 (0.34947187500000004, 0.8246919642857142),
 (0.65343645833333330, 0.8246919642857142)]
```

To align the face, use a landmarks regression model: using regressed points and the given reference landmarks, build an affine transformation to transform regressed points to the reference ones and apply this transformation to the input face image.

## Inputs

Input image, name: `0` , shape: `1, 3, 128, 128` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs
### 1. Intel original
The net outputs a blob with the shape `1, 256, 1, 1`, containing a row-vector of 256 floating point values. Outputs on different images are comparable in cosine distance.
### 2. PINTO custom
1. Model without post-processing

    The net outputs with the shape `1, 256` or `N, 256`, containing a row-vector of 256 floating point values. Outputs on different images are comparable in cosine distance.

2. Model with post-processing

    The net outputs with the shape `N, 256 (feature vector)` and `N, M (cosine distance)`.

    ![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/7949432d-95ae-46d1-83be-88266b945716)


## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Face Recognition Python\* Demo](../../../demos/face_recognition_demo/python/README.md)
* [Smart Classroom C++ Demo](../../../demos/smart_classroom_demo/cpp/README.md)
* [Smart Classroom C++ G-API Demo](../../../demos/smart_classroom_demo/cpp_gapi/README.md)

## Validation

```bash
python demo/validation.py
```

|Pattern|Base image|Target image|Pattern|Base image|Target image|
|:-:|:-:|:-:|:-:|:-:|:-:|
|0 vs 1|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0001](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/aaba96d9-3d2f-4ecf-a6ad-858b80c0ea22)|0 vs 6|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0006](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/01c77b03-3435-49fb-a3c9-a1a7807cec04)|
|0 vs 2|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0002](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/4db0e5dc-afd7-4d97-a6c0-0c519447c8ff)|0 vs 7|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0007](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/cde8eb0b-c356-48da-ab26-fc49cdff7b0b)|
|0 vs 3|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0003](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8affc28f-c0d0-4c87-87db-3aafa759f4a6)|0 vs 8|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0008](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/70208791-eb52-4701-b532-4b72fe8defb4)|
|0 vs 4|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0004](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/438f84d7-56e8-4e07-9b56-26d9d5cac990)|0 vs 9|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0009](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/d79eaca0-8178-4ac9-a6f6-238137344608)|
|0 vs 5|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0005](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/f22d41c5-4b11-4bf1-ab96-22792fa33054)|0 vs 0|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/65d639b6-d2ff-4684-9ad2-ba048380b603)|![12_Group_Group_12_Group_Group_12_10_0000](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/9d16227f-8b70-4950-a584-fb86617fa252)|

|Base|1|2|3|4|5|6|7|8|9|0|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|**0**|0.243|0.029|0.000|0.000|0.000|0.047|0.337|0.181|0.022|1.000|

## Legal Information
[*] Other names and brands may be claimed as the property of others.
