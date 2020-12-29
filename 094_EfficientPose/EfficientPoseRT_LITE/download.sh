#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hO9l7pN7lY1Dxxcd3OR3lHgw-GSj8iEq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hO9l7pN7lY1Dxxcd3OR3lHgw-GSj8iEq" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
