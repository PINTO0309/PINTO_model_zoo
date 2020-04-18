#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1CAuJ34W34LkD20IHSHVx-Z-BzwdEqCsE" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1CAuJ34W34LkD20IHSHVx-Z-BzwdEqCsE" -o deeplab_mnv3_small_cityscapes_trainfine.tflite

echo Download finished.
