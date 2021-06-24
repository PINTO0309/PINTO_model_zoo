#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1S8mKC55jTaNSyHs2e0J8BPDia9jd5SSi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1S8mKC55jTaNSyHs2e0J8BPDia9jd5SSi" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
