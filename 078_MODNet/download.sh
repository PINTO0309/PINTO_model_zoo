#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1bs0Tu0Tu_zIqp-ZCVNW1M6M8_c7kw4B3" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1bs0Tu0Tu_zIqp-ZCVNW1M6M8_c7kw4B3" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
