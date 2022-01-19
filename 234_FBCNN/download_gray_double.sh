#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1i8eUGO9S7eMR6jCqm9GAwuiKW0OID1OK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1i8eUGO9S7eMR6jCqm9GAwuiKW0OID1OK" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
