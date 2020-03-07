#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13anvz7__izd0NsM-30r5-c4YmYyq-yYZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13anvz7__izd0NsM-30r5-c4YmYyq-yYZ" -o efficientnet-lite1.tar.gz
tar -zxvf efficientnet-lite1.tar.gz
rm efficientnet-lite1.tar.gz
echo Download finished.
