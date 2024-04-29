#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/034_ssd_mobilenet_v2_mnasfpn_shared_box_predictor/034_ssd_mobilenet_v2_mnasfpn_shared_box_predictor.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
