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

#### T-ain-aug - OSNet-AIN x1.5 - 5.1M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | T-ain-aug | OSNet-AIN x1.5<br>+<br>cam-branch | 5.1M | 2.12 | 512 | 89.9 | 95.4 | 97.6 | 98.2 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9736 | 0.9869 | 0.9967 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.8863 | 0.9527 | 0.9791 | 0.9827 |
  | duke_occ | 2,210 | 17,661 | 0.9116 | 0.9416 | 0.9769 | 0.9810 |
  | cuhk03np | 1,400 | 5,332 | 0.9801 | 0.9814 | 0.9936 | 0.9971 |
  | occ_reid | 1,000 | 1,000 | 0.9876 | 0.9900 | 0.9930 | 0.9970 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9160 | 0.9613 | — | — |
  | bright+30% | 0.9133 | 0.9602 | -0.0026 | -0.0011 |
  | dark-30% | 0.9148 | 0.9601 | -0.0012 | -0.0012 |
  | contrast-40% | 0.9160 | 0.9610 | +0.0000 | -0.0003 |
  | contrast+40% | 0.8308 | 0.8874 | -0.0852 | -0.0739 |
  | warm | 0.8400 | 0.9003 | -0.0760 | -0.0610 |
  | cool | 0.8586 | 0.9171 | -0.0574 | -0.0442 |
  | gamma0.6 | 0.8994 | 0.9495 | -0.0165 | -0.0118 |
  | gamma1.6 | 0.8843 | 0.9402 | -0.0317 | -0.0211 |
  | jpeg-q40 | 0.9094 | 0.9568 | -0.0066 | -0.0045 |
  | jpeg-q20 | 0.8902 | 0.9418 | -0.0258 | -0.0195 |

#### N-ain-aug - OSNet-AIN x1.25 - 3.8M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | N-ain-aug | OSNet-AIN x1.25<br>+<br>cam-branch | 3.8M | 1.49 | 512 | 90.0 | 95.7 | 97.7 | 98.3 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9731 | 0.9852 | 0.9947 | 0.9982 |
  | msmt17 | 11,659 | 82,161 | 0.8831 | 0.9512 | 0.9781 | 0.9828 |
  | duke_occ | 2,210 | 17,661 | 0.9082 | 0.9425 | 0.9760 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9799 | 0.9807 | 0.9936 | 0.9971 |
  | occ_reid | 1,000 | 1,000 | 0.9878 | 0.9910 | 0.9940 | 0.9990 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9136 | 0.9602 | — | — |
  | bright+30% | 0.9106 | 0.9570 | -0.0030 | -0.0032 |
  | dark-30% | 0.9126 | 0.9592 | -0.0010 | -0.0010 |
  | contrast-40% | 0.9136 | 0.9600 | -0.0000 | -0.0002 |
  | contrast+40% | 0.8302 | 0.8880 | -0.0835 | -0.0722 |
  | warm | 0.8394 | 0.9005 | -0.0742 | -0.0596 |
  | cool | 0.8617 | 0.9214 | -0.0520 | -0.0388 |
  | gamma0.6 | 0.8991 | 0.9507 | -0.0146 | -0.0095 |
  | gamma1.6 | 0.8827 | 0.9399 | -0.0310 | -0.0203 |
  | jpeg-q40 | 0.9072 | 0.9554 | -0.0064 | -0.0048 |
  | jpeg-q20 | 0.8877 | 0.9400 | -0.0259 | -0.0202 |

#### P-ain-aug - OSNet-AIN x1.0 - 2.7M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | P-ain-aug | OSNet-AIN x1.0<br>+<br>cam-branch | 2.7M | 0.98 | 512 | 89.8 | 95.8 | 97.8 | 98.2 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9733 | 0.9881 | 0.9952 | 0.9973 |
  | msmt17 | 11,659 | 82,161 | 0.8832 | 0.9511 | 0.9784 | 0.9827 |
  | duke_occ | 2,210 | 17,661 | 0.9031 | 0.9321 | 0.9724 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9804 | 0.9850 | 0.9929 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9889 | 0.9890 | 0.9940 | 0.9980 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9132 | 0.9597 | — | — |
  | bright+30% | 0.9102 | 0.9576 | -0.0030 | -0.0021 |
  | dark-30% | 0.9124 | 0.9588 | -0.0007 | -0.0009 |
  | contrast-40% | 0.9132 | 0.9596 | +0.0000 | -0.0001 |
  | contrast+40% | 0.8325 | 0.8887 | -0.0807 | -0.0709 |
  | warm | 0.8458 | 0.9060 | -0.0674 | -0.0536 |
  | cool | 0.8619 | 0.9227 | -0.0513 | -0.0370 |
  | gamma0.6 | 0.8986 | 0.9499 | -0.0146 | -0.0097 |
  | gamma1.6 | 0.8839 | 0.9392 | -0.0293 | -0.0204 |
  | jpeg-q40 | 0.9068 | 0.9548 | -0.0064 | -0.0049 |
  | jpeg-q20 | 0.8873 | 0.9405 | -0.0259 | -0.0192 |

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
