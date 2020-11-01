#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xd-yMgKHm55qMbTv0g-LJq41U6xVZGai" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xd-yMgKHm55qMbTv0g-LJq41U6xVZGai" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
