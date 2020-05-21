#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14L3C9YGvnnCv4HfyuZhJjBKb_3A4Zxap" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14L3C9YGvnnCv4HfyuZhJjBKb_3A4Zxap" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
