#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RXOC04BcYwiBDqw1vGn3r2gl5GQ0-O49" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RXOC04BcYwiBDqw1vGn3r2gl5GQ0-O49" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
