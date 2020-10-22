#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14uRDhddRaXtlV1K83p0CTozdDYi6nu0O" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14uRDhddRaXtlV1K83p0CTozdDYi6nu0O" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
