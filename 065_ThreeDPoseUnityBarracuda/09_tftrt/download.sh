#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qFZi6fguw0_RY-tHN7zgQ45xvEUEtQYN" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qFZi6fguw0_RY-tHN7zgQ45xvEUEtQYN" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
