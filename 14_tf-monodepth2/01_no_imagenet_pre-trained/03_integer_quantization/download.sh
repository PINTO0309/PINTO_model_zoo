#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1aUnn0WVDld8XItk0i5HgKEULPRrHhPQU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1aUnn0WVDld8XItk0i5HgKEULPRrHhPQU" -o monodepth2_colormap_depth_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jZ9rWmLGTonwnGhjQLO9jPMFjpmR9YG7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jZ9rWmLGTonwnGhjQLO9jPMFjpmR9YG7" -o monodepth2_colormap_only_integer_quant.tflite

echo Download finished.
