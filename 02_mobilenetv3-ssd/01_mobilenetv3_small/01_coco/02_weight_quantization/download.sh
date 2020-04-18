#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qWiKax3QMvPOtZme-Aqglw_nRxQ1cpAX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qWiKax3QMvPOtZme-Aqglw_nRxQ1cpAX" -o ssd_mobilenet_v3_small_coco_weight_quant.tflite

echo Download finished.
