#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1FjwPWbBND1b4Ppz3GOCox_DtJc69Ubsi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1FjwPWbBND1b4Ppz3GOCox_DtJc69Ubsi" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
