#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1AUWcRc18F1As4u34ti9sUcYeZhjUGNIX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1AUWcRc18F1As4u34ti9sUcYeZhjUGNIX" -o monodepth2_colormap_depth_weight_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=10apjPIBJcOLvyrwRiFZcRLwZQj8ZgD0a" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=10apjPIBJcOLvyrwRiFZcRLwZQj8ZgD0a" -o monodepth2_colormap_only_weight_quant.tflite

echo Download finished.

