#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mg2y7mukVyZjS3YU8Q4g9SnbaoufvX2J" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mg2y7mukVyZjS3YU8Q4g9SnbaoufvX2J" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
