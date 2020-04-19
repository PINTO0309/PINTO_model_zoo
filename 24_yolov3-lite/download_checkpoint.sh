#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1f0OOcM1g-v5WMBtwTvGfsQ5u5eUdyXht" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1f0OOcM1g-v5WMBtwTvGfsQ5u5eUdyXht" -o checkpoint.tar.gz
tar -zxvf checkpoint.tar.gz
rm checkpoint.tar.gz
echo Download finished.
