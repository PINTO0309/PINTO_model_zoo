#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ekUIpdtKbtRVm3-LL20dQWklz-_PmcHa" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ekUIpdtKbtRVm3-LL20dQWklz-_PmcHa" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
