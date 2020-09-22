#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DNX5s8NfH2A4GlpF8wA6AXo8RtXC75Bh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DNX5s8NfH2A4GlpF8wA6AXo8RtXC75Bh" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
