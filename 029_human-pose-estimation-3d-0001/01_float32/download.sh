#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1q866mY4lsB2qz50K-FanaFxw9s2Ps7Y5" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1q866mY4lsB2qz50K-FanaFxw9s2Ps7Y5" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
