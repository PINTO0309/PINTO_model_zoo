#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jg1154fnnVHSs0hhrmJGzBSqVd5ZSRCS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jg1154fnnVHSs0hhrmJGzBSqVd5ZSRCS" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
