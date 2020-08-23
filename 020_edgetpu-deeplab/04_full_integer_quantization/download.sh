#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17FL8i7O61lShdOtjvB95t7mR_UXZ5nr-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17FL8i7O61lShdOtjvB95t7mR_UXZ5nr-" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
