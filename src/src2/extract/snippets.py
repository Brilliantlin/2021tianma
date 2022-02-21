import numpy as np
import os, sys
import jieba
from bert4keras.snippets import open

jieba.initialize()
# 设置递归深度
sys.setrecursionlimit(1000000)

# 标注数据
data_json = '../datasets/train.json'

# 保存权重的文件夹
if not os.path.exists('weights'):
    os.mkdir('weights')

# bert配置
config_path = '../../FinBERT_L-12_H-768_A-12_tf/bert_config.json'
checkpoint_path = '../../FinBERT_L-12_H-768_A-12_tf/bert_model.ckpt'
dict_path = '../../FinBERT_L-12_H-768_A-12_tf/vocab.txt'

# 将数据划分N份，一份作为验证集
num_folds = 15

def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D


