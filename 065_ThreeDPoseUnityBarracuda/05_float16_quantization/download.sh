#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ikHjTmrFMagu1v11Rp3iD8ye6_9W-LXh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ikHjTmrFMagu1v11Rp3iD8ye6_9W-LXh" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
