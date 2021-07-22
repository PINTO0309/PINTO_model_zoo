#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Fkh3N5fhyvrkWBF-9X7FhsN6YveFFu_O" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Fkh3N5fhyvrkWBF-9X7FhsN6YveFFu_O" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
