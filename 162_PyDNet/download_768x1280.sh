#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gfxLTFXku2vak8wjg5jdTDy1gTy1vbUH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gfxLTFXku2vak8wjg5jdTDy1gTy1vbUH" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
