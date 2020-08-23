#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Nf4_eLd2e24VWAxYPk3_8cw7fSaXM9i4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Nf4_eLd2e24VWAxYPk3_8cw7fSaXM9i4" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
