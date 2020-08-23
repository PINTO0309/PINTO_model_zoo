#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RvQZhjsn5EH6O8vXKy_8KIwr5i-s0ecE" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RvQZhjsn5EH6O8vXKy_8KIwr5i-s0ecE" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
