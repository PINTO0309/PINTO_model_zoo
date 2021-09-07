#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wNn5IRH6W_e4RRpESmhx6_V3IpDqDcDq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wNn5IRH6W_e4RRpESmhx6_V3IpDqDcDq" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
