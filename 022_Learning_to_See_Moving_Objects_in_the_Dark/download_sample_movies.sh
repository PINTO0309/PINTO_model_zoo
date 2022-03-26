#!/bin/bash

fileid="1kxtTDg9H50Onkzgv-mvt8n5hv69lxziW"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o sample_movies.tar.gz
tar -zxvf sample_movies.tar.gz
rm sample_movies.tar.gz
rm cookie

echo Download finished.
