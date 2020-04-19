#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1pNKY16fwvIMC6uUCQXQT8SIz-4u2hjy6" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1pNKY16fwvIMC6uUCQXQT8SIz-4u2hjy6" -o yolov3_lite_voc_256_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Nnvt2ix6u9vx5bFeXd9lALo8inaKsdEW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Nnvt2ix6u9vx5bFeXd9lALo8inaKsdEW" -o yolov3_lite_voc_320_float16_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1VpmJtj3-LyEPpo_m_irOiZs2VJifo3yI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1VpmJtj3-LyEPpo_m_irOiZs2VJifo3yI" -o yolov3_lite_voc_416_float16_quant.tflite

echo Download finished.
