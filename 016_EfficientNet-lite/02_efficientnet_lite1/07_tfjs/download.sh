#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wbQ1Rp2mRje00DzRPNELuUFQFRzBe1pt" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wbQ1Rp2mRje00DzRPNELuUFQFRzBe1pt" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
