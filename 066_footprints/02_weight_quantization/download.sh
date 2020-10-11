#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xpBnJ7kDcwt3icHmBoPX-6yPqM6EC4y_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xpBnJ7kDcwt3icHmBoPX-6yPqM6EC4y_" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
