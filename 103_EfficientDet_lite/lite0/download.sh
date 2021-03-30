#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Q4HbEmkZV821c8mHVtO-HrMDSpAQtN1o" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Q4HbEmkZV821c8mHVtO-HrMDSpAQtN1o" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
