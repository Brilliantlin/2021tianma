#!/bin/bash
source activate paddle
cd src2/extract
echo '开始推理摘要部分......'
pwd
#python infer.py

pwd
echo '开始推理prompt......'
cd ../classify
#python infer_prompt.py

cd ../../src1
pwd
echo '开始推理han......'
#python HAN_infer.py

cd ../
python ensemble.py
