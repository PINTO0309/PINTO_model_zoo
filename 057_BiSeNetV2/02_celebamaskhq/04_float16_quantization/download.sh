#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RK4JTJUL9fA0aPCX1umznev5mHLbHk8S" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RK4JTJUL9fA0aPCX1umznev5mHLbHk8S" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
