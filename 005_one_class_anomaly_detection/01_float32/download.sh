#!/bin/bash

fileid="1DVadju5YcRQzS5JzAFCFxdgwDh6-sifv"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o weights.pb

echo Download finished.
