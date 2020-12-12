#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1axNDHBy5FnELuKRruY6Qk8CLixl3dq9X" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1axNDHBy5FnELuKRruY6Qk8CLixl3dq9X" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
