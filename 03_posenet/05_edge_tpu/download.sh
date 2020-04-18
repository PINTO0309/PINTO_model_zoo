#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1X5aoKDTW0h5dg2BD3u4ciaA2DXrmTNI1" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1X5aoKDTW0h5dg2BD3u4ciaA2DXrmTNI1" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
