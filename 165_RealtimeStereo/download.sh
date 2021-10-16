#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1To3O41da0adm0EKObJ8ig0FvC_ZETcRv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1To3O41da0adm0EKObJ8ig0FvC_ZETcRv" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
