#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1r3AWf2R3joF6Q9mFbxvIEGIT2kZfvq_n" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1r3AWf2R3joF6Q9mFbxvIEGIT2kZfvq_n" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
