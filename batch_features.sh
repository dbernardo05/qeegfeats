#!/bin/bash


python scripts/generate_features.py -n 16 \
  -c 'configs/baruto.yml' \
  -o 'features/' \
  -a 1
