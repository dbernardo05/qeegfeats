#!/bin/bash


python scripts/generate_features.py -n 16 \
  -i '/your/eeg/dir/with/hdf5files/'  \
  -c 'configs/baruto.yml' \
  -o 'features/' \
  -a 1
