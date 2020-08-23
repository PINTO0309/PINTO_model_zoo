#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_3zPqzwb85OKGolI2DBBuVvJKPPm4TSE" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_3zPqzwb85OKGolI2DBBuVvJKPPm4TSE" -o deeplabv3_257_mv_gpu.tflite

echo Download finished.
