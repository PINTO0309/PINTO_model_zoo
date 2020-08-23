#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1m_Qy4qtU1zzzYmcJcaDYGyhqX1Q0N8cd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1m_Qy4qtU1zzzYmcJcaDYGyhqX1Q0N8cd" -o deeplab_mnv3_large_cityscapes_trainfine.tflite

echo Download finished.
