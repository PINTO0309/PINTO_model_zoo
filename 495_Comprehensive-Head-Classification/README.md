# 495_Comprehensive-Head-Classification

Comprehensive head classification. Presence/absence of hats, sunglasses, and masks; eyes open/closed; mouth open/closed; background simplicity/complexity; and Face Image Quality Assessment (FIQA).

It is capable of rapidly performing seven types of classification and inference in a single inference pass.

Merged model inputs:

- `head_image_48x48`: `[1, 3, 48, 48]`, head crop used for background, mask, sunglasses, and hat classification
- `eye_images_24x40`: `[2, 3, 24, 40]`, two eye crops used for eye-open classification
- `mouth_image_30x48`: `[1, 3, 30, 48]`, mouth crop used for mouth-open classification
- `head_image_352x352`: `[1, 3, 352, 352]`, head crop used for FIQA in FIQA-enabled models

Input normalization:

- `head_image_48x48`, `eye_images_24x40`, and `mouth_image_30x48`: RGB `float32`, normalized to `0.0..1.0` by dividing pixel values by `255`
- `head_image_352x352`: RGB `float32`, normalized to `0.0..1.0` and then ImageNet-normalized with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

Merged model outputs:

- `prob_bg_plain`: `[1]`, probability that the background is plain/complicated
- `prob_masked`: `[1]`, probability that the person is wearing a mask
- `prob_sunglass`: `[1]`, probability that the person is wearing sunglasses
- `prob_hat`: `[1]`, probability that the person is wearing a hat
- `prob_eye_open`: `[2]`, probability that each eye is open
- `prob_mouth_open`: `[1]`, probability that the mouth is open
- `quality_score`: `[1, 1]`, face image quality score for FIQA-enabled models

<img width="2101" height="1073" alt="image" src="https://github.com/user-attachments/assets/456599bf-d18a-4d8a-b1be-3d5f88e52514" />

## Demo

https://github.com/PINTO0309/Comprehensive-Head-Classification

<img width="1177" height="1230" alt="image" src="https://github.com/user-attachments/assets/a7ef3c22-3bd0-4abf-9963-cd4e9ca3b57d" />


## Cited - Ultra-lightweight classification model series

1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License
9. [MWC: Mask wearing classifier.](https://github.com/PINTO0309/MWC) - MIT License
10. [SGC: Classification of wearing vs. not wearing sunglasses. 48x48.](https://github.com/PINTO0309/SGC) - MIT License
11. [HHC: Head Hat Classification. HHC is a binary classifier for cropped head images. 48x48.](https://github.com/PINTO0309/HHC) - MIT License
