#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Plwwp4Oq9d-X643Sn21GpXY5F_irU3lR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Plwwp4Oq9d-X643Sn21GpXY5F_irU3lR" -o mobilenet_v2_0.5_224_frozen.pb

echo Download finished.
