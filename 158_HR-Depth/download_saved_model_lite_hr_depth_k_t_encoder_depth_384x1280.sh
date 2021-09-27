#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wTkTB81_SkTF2R00i-qxtPGR_b365l8S" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wTkTB81_SkTF2R00i-qxtPGR_b365l8S" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
