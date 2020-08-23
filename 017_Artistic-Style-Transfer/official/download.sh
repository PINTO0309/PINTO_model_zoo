#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1X_YS6Mr3Lmo9sLfCbVHa-RaKlni9FxF7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1X_YS6Mr3Lmo9sLfCbVHa-RaKlni9FxF7" -o style_predict_quantized_256.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mGwTGI1bcuTe4fUFGBvwr9AMGN0HcOzn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mGwTGI1bcuTe4fUFGBvwr9AMGN0HcOzn" -o style_transfer_quantized_dynamic.tflite

echo Download finished.
