# 502_PersonViT

PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification

## Results
<img width="439" height="412" alt="image" src="https://github.com/user-attachments/assets/8ddee91a-8aa9-426d-a7f6-37615792d089" />

<img width="1606" height="544" alt="image" src="https://github.com/user-attachments/assets/5c63d6d2-ba6e-489b-a632-a0f8238eff08" />


## Comparison with OSNet

[OSNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html) is a useful lightweight CNN baseline for evaluating the accuracy/efficiency trade-off of PersonViT. A practical comparison should include OSNet x1.0, PersonViT-S/16, and PersonViT-B/16. For deployment without target-domain adaptation, [OSNet-AIN x1.0](https://arxiv.org/abs/1910.06827) is also relevant because it was designed for improved cross-domain generalization.

| Model | Parameters | GFLOPs at 256 x 128 | Embedding dimension | Typical use |
| --- | ---: | ---: | ---: | --- |
| OSNet x1.0 | 2.2M | 0.98 | 512 | Edge or real-time deployment |
| PersonViT-S/16 | 22.0M | 2.94 | 384 | Balanced accuracy and inference cost |
| PersonViT-B/16 | 86.5M | 11.35 | 768 | Accuracy-oriented GPU deployment |

The OSNet complexity values are from the [official Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html). The PersonViT values are calculated from the models in this repository with a 256 x 128 input and a 16 x 16 stride. Runtime latency can differ from FLOPs and must be measured with the target hardware and inference backend.

The following published checkpoint results are shown as **Rank-1 / mAP** and are useful as an off-the-shelf reference:

| Model | Market1501 | DukeMTMC-reID | MSMT17 |
| --- | ---: | ---: | ---: |
| OSNet x1.0 | 94.2 / 82.6 | 87.0 / 70.2 | 74.9 / 43.8 |
| PersonViT-S/16 | 96.8 / 92.9 | 91.9 / 84.7 | 88.8 / 74.3 |
| PersonViT-B/16 | 97.6 / 95.0 | 93.8 / 88.1 | 92.0 / 80.8 |

OSNet results are from the [official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html), and PersonViT results are from the [released fine-tuning logs](https://huggingface.co/lakeAGI/PersonViTReID). These numbers are **not a controlled architecture-only comparison**: the released models use different pre-training, loss functions, optimizers, data augmentation, and evaluation details. In particular, the OSNet model-zoo baseline uses softmax loss, while PersonViT uses softmax and triplet losses.

For a fair deployment comparison:

1. Use exactly the same person detections, crops, query/gallery split, and 256 x 128 input resolution.
2. Keep each model's expected pixel normalization; OSNet uses ImageNet normalization, while the released PersonViT configuration uses mean and standard deviation `[0.5, 0.5, 0.5]`.
3. L2-normalize both embeddings, use the same distance metric, and disable re-ranking.
4. Measure mAP, Rank-1, false-accept and false-reject rates, batch-1 p50/p95 latency, throughput, peak memory, and model size.
5. Use the same hardware, precision (FP32 or FP16), runtime, batch size, and warm-up procedure.
6. Report results separately for operational conditions such as occlusion, low resolution, nighttime, and individual cameras.

As a starting point, prefer OSNet x1.0 when latency and memory are the primary constraints, PersonViT-S/16 when a moderate compute budget is available, and PersonViT-B/16 when retrieval accuracy is the priority. The final choice should be based on validation data collected from the actual deployment environment.

---
---
---

## August 16, 2026: Significantly enhanced generalization performance

- Preprocessing: resize the RGB person crop to 256 x 128, scale pixels to `[0, 1]`, then normalize each channel with mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]`.
- Output embeddings are L2 normalized.

Example ONNX Runtime inference:

```python
import cv2
import numpy as np
import onnxruntime as ort

image = cv2.imread("person.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 256)).astype(np.float32) / 255.0
image = (image - 0.5) / 0.5
images = np.transpose(image, (2, 0, 1))[None]

session = ort.InferenceSession(
    "personvit_vitb16_ain_unified_aug_n.onnx",
    providers=["CPUExecutionProvider"],
)
embeddings = session.run(["embeddings"], {"images": images})[0]
```

#### B-ain-aug -  ViT-B/16 + token-IN - 86.5M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | B-ain-aug | ViT-B/16<br>+<br>token-IN | 86.5M | 11.35 | 768 | 93.6 | 96.8 | 98.1 | 98.5 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9905 | 0.9941 | 0.9982 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.9526 | 0.9738 | 0.9885 | 0.9901 |
  | duke_occ | 2,210 | 17,661 | 0.9584 | 0.9697 | 0.9828 | 0.9860 |
  | cuhk03np | 1,400 | 5,332 | 0.9891 | 0.9893 | 0.9950 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9961 | 0.9970 | 0.9980 | 1.0000 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9646 | 0.9791 | — | — |
  | bright+30% | 0.9640 | 0.9786 | -0.0006 | -0.0006 |
  | dark-30% | 0.9646 | 0.9792 | +0.0000 | +0.0001 |
  | contrast-40% | 0.9646 | 0.9791 | +0.0000 | +0.0000 |
  | contrast+40% | 0.9253 | 0.9461 | -0.0393 | -0.0330 |
  | warm | 0.9351 | 0.9546 | -0.0295 | -0.0245 |
  | cool | 0.9515 | 0.9698 | -0.0131 | -0.0094 |
  | gamma0.6 | 0.9585 | 0.9745 | -0.0061 | -0.0046 |
  | gamma1.6 | 0.9561 | 0.9743 | -0.0085 | -0.0048 |
  | jpeg-q40 | 0.9621 | 0.9773 | -0.0025 | -0.0018 |
  | jpeg-q20 | 0.9549 | 0.9731 | -0.0097 | -0.0061 |

#### S-ain-aug -  ViT-S/16 + token-IN - 22.0M

- [CrowdTrack](https://github.com/loseevaya/CrowdTrack) dataset test - Testing using video footage that was not included in the training data at all.

  https://github.com/user-attachments/assets/821f4812-cfd5-402c-b4b5-fa5776ed1ce1
  
  https://github.com/user-attachments/assets/baf48f48-d22b-4d79-b2de-44cdb8b2a967

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | S-ain-aug | ViT-S/16<br>+<br>token-IN | 22.0M | 2.94 | 384 | 93.1 | 97.2 | 98.2 | 98.4 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9872 | 0.9911 | 0.9976 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.9397 | 0.9697 | 0.9860 | 0.9882 |
  | duke_occ | 2,210 | 17,661 | 0.9527 | 0.9674 | 0.9819 | 0.9855 |
  | cuhk03np | 1,400 | 5,332 | 0.9875 | 0.9900 | 0.9950 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9955 | 0.9960 | 0.9980 | 0.9990 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9555 | 0.9759 | — | — |
  | bright+30% | 0.9547 | 0.9753 | -0.0008 | -0.0006 |
  | dark-30% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast-40% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast+40% | 0.9090 | 0.9400 | -0.0465 | -0.0360 |
  | warm | 0.9192 | 0.9477 | -0.0363 | -0.0282 |
  | cool | 0.9411 | 0.9664 | -0.0145 | -0.0095 |
  | gamma0.6 | 0.9481 | 0.9717 | -0.0074 | -0.0042 |
  | gamma1.6 | 0.9448 | 0.9705 | -0.0107 | -0.0054 |
  | jpeg-q40 | 0.9523 | 0.9748 | -0.0032 | -0.0011 |
  | jpeg-q20 | 0.9430 | 0.9693 | -0.0125 | -0.0066 |

#### T-ain-aug - OSNet-AIN x1.5 - 4.6M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | T-ain-aug | OSNet-AIN x1.5 | 4.6M | 2.12 | 512 | 88.9 | 95.0 | 97.4 | 98.2 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9722 | 0.9875 | 0.9958 | 0.9979 |
  | msmt17 | 11,659 | 82,161 | 0.8731 | 0.9476 | 0.9768 | 0.9824 |
  | duke_occ | 2,210 | 17,661 | 0.9115 | 0.9421 | 0.9751 | 0.9828 |
  | cuhk03np | 1,400 | 5,332 | 0.9811 | 0.9836 | 0.9936 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9842 | 0.9870 | 0.9910 | 0.9970 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9078 | 0.9584 | — | — |
  | bright+30% | 0.9041 | 0.9560 | -0.0037 | -0.0024 |
  | dark-30% | 0.9066 | 0.9578 | -0.0012 | -0.0006 |
  | contrast-40% | 0.9078 | 0.9582 | -0.0000 | -0.0002 |
  | contrast+40% | 0.8149 | 0.8751 | -0.0929 | -0.0833 |
  | warm | 0.8044 | 0.8712 | -0.1034 | -0.0872 |
  | cool | 0.8247 | 0.8903 | -0.0831 | -0.0681 |
  | gamma0.6 | 0.8833 | 0.9415 | -0.0245 | -0.0169 |
  | gamma1.6 | 0.8722 | 0.9349 | -0.0356 | -0.0235 |
  | jpeg-q40 | 0.9005 | 0.9533 | -0.0073 | -0.0051 |
  | jpeg-q20 | 0.8801 | 0.9376 | -0.0277 | -0.0208 |

#### N-ain-aug - OSNet-AIN x1.25 - 3.3M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | N-ain-aug | OSNet-AIN x1.25 | 3.3M | 1.49 | 512 | 89.1 | 95.3 | 97.5 | 98.1 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9710 | 0.9860 | 0.9964 | 0.9982 |
  | msmt17 | 11,659 | 82,161 | 0.8710 | 0.9466 | 0.9766 | 0.9818 |
  | duke_occ | 2,210 | 17,661 | 0.9057 | 0.9317 | 0.9760 | 0.9814 |
  | cuhk03np | 1,400 | 5,332 | 0.9805 | 0.9843 | 0.9921 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9835 | 0.9880 | 0.9930 | 0.9980 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9056 | 0.9565 | — | — |
  | bright+30% | 0.9022 | 0.9543 | -0.0035 | -0.0021 |
  | dark-30% | 0.9047 | 0.9564 | -0.0009 | -0.0001 |
  | contrast-40% | 0.9056 | 0.9564 | +0.0000 | -0.0001 |
  | contrast+40% | 0.8153 | 0.8772 | -0.0903 | -0.0793 |
  | warm | 0.8103 | 0.8786 | -0.0953 | -0.0779 |
  | cool | 0.8405 | 0.9057 | -0.0651 | -0.0507 |
  | gamma0.6 | 0.8863 | 0.9443 | -0.0193 | -0.0122 |
  | gamma1.6 | 0.8720 | 0.9330 | -0.0336 | -0.0235 |
  | jpeg-q40 | 0.8989 | 0.9528 | -0.0067 | -0.0036 |
  | jpeg-q20 | 0.8784 | 0.9360 | -0.0272 | -0.0204 |

#### P-ain-aug - OSNet-AIN x1.0 - 2.2M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | P-ain-aug | OSNet-AIN x1.0 | 2.2M | 0.98 | 512 | 89.1 | 95.4 | 97.7 | 98.1 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9711 | 0.9857 | 0.9964 | 0.9976 |
  | msmt17 | 11,659 | 82,161 | 0.8711 | 0.9472 | 0.9780 | 0.9828 |
  | duke_occ | 2,210 | 17,661 | 0.8995 | 0.9362 | 0.9733 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9800 | 0.9857 | 0.9936 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9878 | 0.9900 | 0.9940 | 0.9980 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9051 | 0.9575 | — | — |
  | bright+30% | 0.9014 | 0.9547 | -0.0037 | -0.0027 |
  | dark-30% | 0.9042 | 0.9566 | -0.0009 | -0.0009 |
  | contrast-40% | 0.9051 | 0.9575 | -0.0000 | +0.0001 |
  | contrast+40% | 0.8172 | 0.8775 | -0.0879 | -0.0800 |
  | warm | 0.8159 | 0.8849 | -0.0892 | -0.0726 |
  | cool | 0.8447 | 0.9107 | -0.0605 | -0.0467 |
  | gamma0.6 | 0.8865 | 0.9451 | -0.0186 | -0.0124 |
  | gamma1.6 | 0.8713 | 0.9345 | -0.0338 | -0.0230 |
  | jpeg-q40 | 0.8982 | 0.9529 | -0.0070 | -0.0046 |
  | jpeg-q20 | 0.8776 | 0.9367 | -0.0275 | -0.0208 |

#### osnet_ain_ms_d_c - 2.2M

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.4580 | 0.7304 | 0.8655 | 0.9047 |
  | msmt17 | 11,659 | 82,161 | 0.4869 | 0.7613 | 0.8662 | 0.8965 |
  | duke_occ | 2,210 | 17,661 | 0.4757 | 0.6167 | 0.7670 | 0.8163 |
  | cuhk03np | 1,400 | 5,332 | 0.5776 | 0.6079 | 0.7779 | 0.8543 |
  | occ_reid | 1,000 | 1,000 | 0.7407 | 0.8040 | 0.8970 | 0.9320 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.5001 | 0.7310 | — | — |
  | bright+30% | 0.4945 | 0.7236 | -0.0055 | -0.0074 |
  | dark-30% | 0.4992 | 0.7307 | -0.0009 | -0.0003 |
  | contrast-40% | 0.4903 | 0.7220 | -0.0097 | -0.0090 |
  | contrast+40% | 0.4101 | 0.6127 | -0.0900 | -0.1183 |
  | warm | 0.4159 | 0.6370 | -0.0841 | -0.0940 |
  | cool | 0.4340 | 0.6668 | -0.0661 | -0.0642 |
  | gamma0.6 | 0.4791 | 0.7068 | -0.0210 | -0.0242 |
  | gamma1.6 | 0.4492 | 0.6788 | -0.0508 | -0.0521 |
  | jpeg-q40 | 0.4819 | 0.7087 | -0.0182 | -0.0223 |
  | jpeg-q20 | 0.4379 | 0.6571 | -0.0621 | -0.0738 |

### `-ain` variants vs the standard ladder

Every tier exists (or is planned) in two flavors that share the same training recipe, data and evaluation protocol; the only difference is where the network normalizes:

| | Standard ladder (B/S/T/N/P/F/A) | `-ain` ladder (B-ain, S-ain, ...) |
| --- | --- | --- |
| Normalization | BatchNorm (CNN) / LayerNorm (ViT) only | Adds instance normalization at style-sensitive early positions: token-axis IN after the ViT patch embedding; the searched OSNet-AIN placement (IN stem + four IN blocks) for CNN tiers |
| What the IN does | — | Removes each image's own style statistics (illumination, color cast, camera tone) from the features at inference time |
| In-distribution accuracy | Highest on the unified test set | Slightly lower by design |
| Unseen-environment robustness | Sensitive to camera/style shift; BatchNorm also carries training-set statistics into deployment | Style-invariant features and per-sample normalization — measured with the style-shift probe: mean mAP drop over 8 photometric shifts falls from 5.2 to 3.4 (B pair), 5.3 to 3.9 (S pair), 10.9 to 5.8 (N pair) and 10.3 to 5.1 (P pair), with exact-zero degradation under uniform gain/contrast shifts; under the hardest shift the `-ain` models beat their BN siblings in absolute mAP despite the lower clean score; photometric-augmentation fine-tunes further cut the mean drop (B: 1.8, S: 2.2, P: 3.5, N: 3.8, T: 3.9) while also raising clean mAP (B: 92.3, S: 91.6, P: 87.8, N: 88.4, T: 88.6) |
| Teacher for distilled tiers | B | B-ain |
| ONNX | BatchNorm folds away entirely | InstanceNormalization nodes remain (runtime normalization; ViT: 1 node, OSNet: 5) with a small latency overhead |

Choose the standard ladder when the deployment cameras resemble the training domains, and the `-ain` ladder when deploying to new environments without target-domain fine-tuning.
