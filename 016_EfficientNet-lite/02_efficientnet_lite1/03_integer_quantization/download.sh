#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1s7moT8F0oMPWTwwKNd9iM6XfwqsqK0if" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1s7moT8F0oMPWTwwKNd9iM6XfwqsqK0if" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
