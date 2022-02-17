#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lw1ZeHgB8W_yukoQGUzUjFUXfo14T1ao" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lw1ZeHgB8W_yukoQGUzUjFUXfo14T1ao" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
