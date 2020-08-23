#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RC3uWAqaHm5-Xzj6YbyM8xeQmcwD50TR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RC3uWAqaHm5-Xzj6YbyM8xeQmcwD50TR" -o weights_integer_quant.tflite

echo Download finished.
