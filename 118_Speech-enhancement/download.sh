#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jbTQKj0G2sW9-2pdwf8P3p8oZAcplagP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jbTQKj0G2sW9-2pdwf8P3p8oZAcplagP" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
