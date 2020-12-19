#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ac2puxcdQQnG91Ye_oC09Zw_XqZjmQ_I" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ac2puxcdQQnG91Ye_oC09Zw_XqZjmQ_I" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
