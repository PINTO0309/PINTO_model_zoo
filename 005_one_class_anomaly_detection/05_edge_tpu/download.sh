#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HRu3Ua2ra1LOgJFGkBDNFBRv5gT94m54" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HRu3Ua2ra1LOgJFGkBDNFBRv5gT94m54" -o weights_full_integer_quant_edgetpu.tflite

echo Download finished.
