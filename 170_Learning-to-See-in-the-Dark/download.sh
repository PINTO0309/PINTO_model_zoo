#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18YgcU70t-IJElieW0JLOfbS3dwCg5qV1" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18YgcU70t-IJElieW0JLOfbS3dwCg5qV1" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
