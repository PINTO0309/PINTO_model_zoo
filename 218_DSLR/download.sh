#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Q7yr2oXCpBCrljlZ7uRR67c7lK3gmrY7" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Q7yr2oXCpBCrljlZ7uRR67c7lK3gmrY7" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
