#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xm5vizWif56d1eBqu81ziJtkgmEujNj0" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xm5vizWif56d1eBqu81ziJtkgmEujNj0" -o v3-small-minimalistic_224_1.0_float.pb

echo Download finished.
