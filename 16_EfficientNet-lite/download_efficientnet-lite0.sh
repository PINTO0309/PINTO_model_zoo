#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1aZkmlu3bA7p-U96TzCdoV72WH7hTVMA4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1aZkmlu3bA7p-U96TzCdoV72WH7hTVMA4" -o efficientnet-lite0.tar.gz
tar -zxvf efficientnet-lite0.tar.gz
rm efficientnet-lite0.tar.gz
echo Download finished.
