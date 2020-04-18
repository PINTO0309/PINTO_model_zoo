#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1iaAfhtvKY76URdJ38qNNc9_VR-KpIDnp" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1iaAfhtvKY76URdJ38qNNc9_VR-KpIDnp" -o deeplabv3_mnv2_dm05_pascal_trainaug_weight_quant.tflite

echo Download finished.
