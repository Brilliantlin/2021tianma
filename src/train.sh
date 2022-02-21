cd src2/extract/
echo "摘要数据准备..."
python dataprepare.py
echo "句向量抽取..."
python extract_vectorize.py
echo "摘要模型训练..."
python extract_model.py
echo "摘要训练完毕"

cd ../classify
pwd python train_prompt.py
echo "prompt 训练完毕"

cd ../../
cd src1
pwd
python HAN_model_fin.py
echo "Han 训练完毕"

