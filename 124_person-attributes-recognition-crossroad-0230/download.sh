#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1StJS7Ya7p-ypUCyTtJJEYBywdDrQNX0e" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1StJS7Ya7p-ypUCyTtJJEYBywdDrQNX0e" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
