#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1T0doDX88rhfSy_aPRVl15g3QmU63z25h" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1T0doDX88rhfSy_aPRVl15g3QmU63z25h" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
