#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1cYPnrNQE6JwzzNAM8jR_c6DmcVleOtZn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1cYPnrNQE6JwzzNAM8jR_c6DmcVleOtZn" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
