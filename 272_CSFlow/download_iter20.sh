#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mjYegy8kMVtLn7u2v4D_EVU77ciEqdcD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mjYegy8kMVtLn7u2v4D_EVU77ciEqdcD" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
