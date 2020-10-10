#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1o_eCNZ9gVZ-DfXqmv9wCnlv684_bzxex" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1o_eCNZ9gVZ-DfXqmv9wCnlv684_bzxex" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
