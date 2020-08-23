#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1GAenxYvmPzahzs175Jk_wtt46OW3BiCK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1GAenxYvmPzahzs175Jk_wtt46OW3BiCK" -o anchors.npy

echo Download finished.
