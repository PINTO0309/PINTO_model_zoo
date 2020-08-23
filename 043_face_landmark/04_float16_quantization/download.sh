#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=184Pl5oP0aIwFeNGRSxFp66Rc4lMDwtEU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=184Pl5oP0aIwFeNGRSxFp66Rc4lMDwtEU" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
