#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Uq4z2mTnkDxrpH_icj7KMHtKUneXkRE9" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Uq4z2mTnkDxrpH_icj7KMHtKUneXkRE9" -o efficientnet-lite4.tar.gz

tar -zxvf efficientnet-lite4.tar.gz
rm efficientnet-lite4.tar.gz
echo Download finished.
