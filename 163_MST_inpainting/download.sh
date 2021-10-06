#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1sjW3cu0I-Kt4_s2hZuw2uuuVHuY_EECj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1sjW3cu0I-Kt4_s2hZuw2uuuVHuY_EECj" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
