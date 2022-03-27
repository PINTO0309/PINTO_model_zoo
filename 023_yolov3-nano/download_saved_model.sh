#!/bin/bash

fileid="11pbvvFJLAS7qLv68odP80Ea6CYkjlQGi"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o saved_model.tar.gz
tar -zxvf saved_model.tar.gz
rm saved_model.tar.gz
echo Download finished.
