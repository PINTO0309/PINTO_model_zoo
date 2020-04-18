#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18_-BmmJx7iP_Y-MJM-0CaB5qzH2m2dTD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18_-BmmJx7iP_Y-MJM-0CaB5qzH2m2dTD" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
