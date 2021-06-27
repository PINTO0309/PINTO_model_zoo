#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1p3Zeyv0vgugjPeg9y03nqdy9o4l9PXjH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1p3Zeyv0vgugjPeg9y03nqdy9o4l9PXjH" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
