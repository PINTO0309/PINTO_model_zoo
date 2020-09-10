#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1p_3R8c5PmgFkxXUe7OWbgFj_tyFn9Zpx" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1p_3R8c5PmgFkxXUe7OWbgFj_tyFn9Zpx" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
