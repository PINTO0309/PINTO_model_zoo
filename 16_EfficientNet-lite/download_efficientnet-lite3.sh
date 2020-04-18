#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gydaW7UTELSSTl3qtwY6etbUWyrj_Ecj" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gydaW7UTELSSTl3qtwY6etbUWyrj_Ecj" -o efficientnet-lite3.tar.gz

tar -zxvf efficientnet-lite3.tar.gz
rm efficientnet-lite3.tar.gz
echo Download finished.
