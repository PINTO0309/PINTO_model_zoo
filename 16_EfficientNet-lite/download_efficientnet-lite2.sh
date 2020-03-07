#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xF8lFS62nofPrJsiKY0AEd9NrdYCezqj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xF8lFS62nofPrJsiKY0AEd9NrdYCezqj" -o efficientnet-lite2.tar.gz
tar -zxvf efficientnet-lite2.tar.gz
rm efficientnet-lite2.tar.gz
echo Download finished.
