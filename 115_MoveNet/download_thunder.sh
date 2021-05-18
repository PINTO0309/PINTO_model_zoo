#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RDF35KcL7kWRb4dgRf0OudH6l0EtZ3qw" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RDF35KcL7kWRb4dgRf0OudH6l0EtZ3qw" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
