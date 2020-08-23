#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1K0hKFxQicJfGMlu4kx0vuPeye6HddiSH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1K0hKFxQicJfGMlu4kx0vuPeye6HddiSH" -o efficientnet_b0_224_weight_quant.tflite

echo Download finished.
