#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11XfaAzv6NK2ENmiUoZUXUEa4QYbb5mxN" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11XfaAzv6NK2ENmiUoZUXUEa4QYbb5mxN" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
