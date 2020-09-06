#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1J1BgS_V_0ASczJ33X2Jt0JEb92ScjRJi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1J1BgS_V_0ASczJ33X2Jt0JEb92ScjRJi" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
