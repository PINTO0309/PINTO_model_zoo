#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1sowZcU-8op1Qt70Prs1MmlVx37lVpHjY" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1sowZcU-8op1Qt70Prs1MmlVx37lVpHjY" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
