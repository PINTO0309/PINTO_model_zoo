#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1k4JZwrHap3cG_3Vt9tQFsahW3Z27-uuh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1k4JZwrHap3cG_3Vt9tQFsahW3Z27-uuh" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
