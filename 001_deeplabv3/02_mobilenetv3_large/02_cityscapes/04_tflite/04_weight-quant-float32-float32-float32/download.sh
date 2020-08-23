#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wwO8hjoilIDDPEwc1acXN7sEINTl3Rgi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wwO8hjoilIDDPEwc1acXN7sEINTl3Rgi" -o deeplab_mnv3_large_weight_quant_257.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1OSKU0ssLzsZYqdMNezAEgSNB0EAdnajz" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1OSKU0ssLzsZYqdMNezAEgSNB0EAdnajz" -o deeplab_mnv3_large_weight_quant_769.tflite

echo Download finished.
