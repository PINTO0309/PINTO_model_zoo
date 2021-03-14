#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1b-id4CvEk8ZtRZRyvu8I_t-3tMivXDmb" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1b-id4CvEk8ZtRZRyvu8I_t-3tMivXDmb" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
