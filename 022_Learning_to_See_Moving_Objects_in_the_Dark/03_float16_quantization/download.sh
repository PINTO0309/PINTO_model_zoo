#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1KkbSYTCm8Z3RxD8V5HccSPY6zQJxXTXv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1KkbSYTCm8Z3RxD8V5HccSPY6zQJxXTXv" -o lsmod_256_float16_quant.tflite

echo Download finished.
