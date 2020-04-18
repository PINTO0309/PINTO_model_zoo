#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1T9IlVn2PWX5bFg4lCe8wd6ZFHRx5XjzA" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1T9IlVn2PWX5bFg4lCe8wd6ZFHRx5XjzA" -o weights_full_integer_quant.tflite

echo Download finished.
