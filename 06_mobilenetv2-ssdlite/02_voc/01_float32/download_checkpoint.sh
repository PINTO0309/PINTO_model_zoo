curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1nus5ZNpzuEwr0ZQqeLdGnE7d3kDMzq1e" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1nus5ZNpzuEwr0ZQqeLdGnE7d3kDMzq1e" -o ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz
tar -zxvf ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz
rm ssdlite_mobilenet_v2_voc_2020_02_04.tar.gz
