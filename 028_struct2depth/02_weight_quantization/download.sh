#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1P-WhmCnFhU-2zansm9QoG-9A1jcN5bC0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1P-WhmCnFhU-2zansm9QoG-9A1jcN5bC0" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
