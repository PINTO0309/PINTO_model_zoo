#!/bin/bash

fileid="1gL7TzJOWAckfwMJKQj8H9t0mOVakKfm7"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o v3-large_224_0.75_float.pb

echo Download finished.
