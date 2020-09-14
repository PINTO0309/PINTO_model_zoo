#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1t3MI4zdShZSx-4q8OAl_n3d5z4tp3VeH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1t3MI4zdShZSx-4q8OAl_n3d5z4tp3VeH" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
