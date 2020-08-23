#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1TrNtsk3bF5hKt4Pqyj33ueNfqmWwIXYX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1TrNtsk3bF5hKt4Pqyj33ueNfqmWwIXYX" -o ssd_mobilenet_v3_small_coco_full_integer_quant.tflite

echo Download finished.
