#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mxm4XAhZY8QKsye2-irSB4MgYmjKVXdC" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mxm4XAhZY8QKsye2-irSB4MgYmjKVXdC" -o lsmod_256_weight_quant.tflite

echo Download finished.
