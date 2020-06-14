#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18XtY_mCOsqnDef3CvBk_B1Y0N49ASVpR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18XtY_mCOsqnDef3CvBk_B1Y0N49ASVpR" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
