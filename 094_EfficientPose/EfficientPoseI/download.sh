#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1zGlv1kxcLF9Rf1WYAP_H26BfYFXk1lZF" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1zGlv1kxcLF9Rf1WYAP_H26BfYFXk1lZF" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
