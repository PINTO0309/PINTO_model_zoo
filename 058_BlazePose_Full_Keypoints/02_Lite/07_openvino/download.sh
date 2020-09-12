#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Ffy3QT8YgrgNlqF8XyfMP7fA9PYLx9nG" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Ffy3QT8YgrgNlqF8XyfMP7fA9PYLx9nG" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
