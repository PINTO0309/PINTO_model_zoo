#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1FB-wB5a_1FhHdQ1RD6OMwxTA1QjHjPXs" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1FB-wB5a_1FhHdQ1RD6OMwxTA1QjHjPXs" -o weights_weight_quant.tflite

echo Download finished.
