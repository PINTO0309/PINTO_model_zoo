#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=152SdlygqIM6hi4JqfnwY5fbK617iQbya" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=152SdlygqIM6hi4JqfnwY5fbK617iQbya" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
