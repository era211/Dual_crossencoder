#!/bin/bash
# download ecb+
CURDIR=data/ecb+


mkdir -p ${CURDIR}/interim
python /home/yaolong/Rationale4CDECR-main/src/data/make_dataset.py

python /home/yaolong/Rationale4CDECR-main/src/features/build_features.py