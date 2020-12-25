#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1zWIdrWVlU0pst_dS2akMPpao1UhyMmeV" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1zWIdrWVlU0pst_dS2akMPpao1UhyMmeV" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
