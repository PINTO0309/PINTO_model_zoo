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

## August 11, 2026: Significantly enhanced generalization performance

### B-ain-aug -  ViT-B/16 + token-IN - 86.5M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.9793 | 0.9875 | 0.9961 | 0.9982 |
| msmt17 | 11,659 | 82,161 | 0.8999 | 0.9557 | 0.9804 | 0.9835 |
| duke_occ | 2,210 | 17,661 | 0.9206 | 0.9471 | 0.9801 | 0.9851 |
| cuhk03np | 1,400 | 5,332 | 0.9795 | 0.9793 | 0.9907 | 0.9964 |
| occ_reid | 1,000 | 1,000 | 0.9970 | 1.0000 | 1.0000 | 1.0000 |

### S-ain-aug -  ViT-S/16 + token-IN - 22.0M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.9752 | 0.9857 | 0.9952 | 0.9979 |
| msmt17 | 11,659 | 82,161 | 0.8886 | 0.9523 | 0.9798 | 0.9829 |
| duke_occ | 2,210 | 17,661 | 0.9089 | 0.9394 | 0.9774 | 0.9860 |
| cuhk03np | 1,400 | 5,332 | 0.9766 | 0.9786 | 0.9914 | 0.9971 |
| occ_reid | 1,000 | 1,000 | 0.9962 | 0.9980 | 1.0000 | 1.0000 |

### T-ain-aug - OSNet-AIN x1.5 - 4.6M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.9684 | 0.9846 | 0.9958 | 0.9976 |
| msmt17 | 11,659 | 82,161 | 0.8669 | 0.9448 | 0.9744 | 0.9799 |
| duke_occ | 2,210 | 17,661 | 0.9031 | 0.9357 | 0.9747 | 0.9810 |
| cuhk03np | 1,400 | 5,332 | 0.9776 | 0.9814 | 0.9936 | 0.9986 |
| occ_reid | 1,000 | 1,000 | 0.9791 | 0.9840 | 0.9890 | 0.9980 |

### N-ain-aug - OSNet-AIN x1.25 - 3.3M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.9673 | 0.9860 | 0.9958 | 0.9982 |
| msmt17 | 11,659 | 82,161 | 0.8609 | 0.9398 | 0.9737 | 0.9803 |
| duke_occ | 2,210 | 17,661 | 0.8922 | 0.9199 | 0.9710 | 0.9805 |
| cuhk03np | 1,400 | 5,332 | 0.9793 | 0.9829 | 0.9936 | 0.9979 |
| occ_reid | 1,000 | 1,000 | 0.9791 | 0.9790 | 0.9910 | 0.9960 |

### P-ain-aug - OSNet-AIN x1.0 - 2.2M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.9642 | 0.9855 | 0.9961 | 0.9979 |
| msmt17 | 11,659 | 82,161 | 0.8537 | 0.9396 | 0.9756 | 0.9807 |
| duke_occ | 2,210 | 17,661 | 0.8837 | 0.9226 | 0.9683 | 0.9787 |
| cuhk03np | 1,400 | 5,332 | 0.9752 | 0.9807 | 0.9936 | 0.9979 |
| occ_reid | 1,000 | 1,000 | 0.9831 | 0.9850 | 0.9940 | 0.9970 |

### osnet_ain_ms_d_c - 2.2M

| dataset | queries | gallery | mAP | R1 | R5 | R10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market | 3,368 | 15,913 | 0.4580 | 0.7304 | 0.8655 | 0.9047 |
| msmt17 | 11,659 | 82,161 | 0.4869 | 0.7613 | 0.8662 | 0.8965 |
| duke_occ | 2,210 | 17,661 | 0.4757 | 0.6167 | 0.7670 | 0.8163 |
| cuhk03np | 1,400 | 5,332 | 0.5776 | 0.6079 | 0.7779 | 0.8543 |
| occ_reid | 1,000 | 1,000 | 0.7407 | 0.8040 | 0.8970 | 0.9320 |

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
