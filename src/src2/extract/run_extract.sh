#!/bin/bash

python dataprepare.py
python extract_vectorize.py
for ((i=0; i<15; i++));
    do
        python extract_model.py $i
    done
