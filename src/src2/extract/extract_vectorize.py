#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 抽取式：句向量化
# 科学空间：https://kexue.fm

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.models import Model
from snippets import *


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载bert模型，补充平均池化
encoder = build_transformer_model(
    config_path,
    checkpoint_path,
)
output = GlobalAveragePooling1D()(encoder.output)
encoder = Model(encoder.inputs, output)


def load_data(filename):
    """加载数据
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append(l[0])
    return D


def predict(texts):
    """句子列表转换为句向量
    """
    batch_token_ids, batch_segment_ids = [], []
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=120)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    outputs = []
    outputs.extend(encoder.predict([batch_token_ids, batch_segment_ids],batch_size=64))
    return outputs

def convert(data):
    """转换所有样本
    """
    embeddings = []
    for texts in tqdm(data, desc=u'向量化'):
        outputs = predict(texts)
        embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


if __name__ == '__main__':

    data_extract_json = data_json[:-5] + '_extract.json'
    data_extract_npy = data_json[:-5] + '_extract3'
    test_extract_json = '../datasets/test_extract.json'
    test_extract_npy = '../datasets/test_extract3.npy'

    test_extract_json_b = '../datasets/test_extract_B.json'
    test_extract_npy_b = '../datasets/test_extract_B.npy'

    #训练集vector 抽取
    data = load_data(data_extract_json) #
    embeddings = convert(data)
    np.save(data_extract_npy, embeddings)
    print(u'输出路径：%s.npy' % data_extract_npy)

    #测试集 A 榜抽取
    # data = load_data(test_extract_json)#[[,,,,],]
    # verctor = convert(data)#[1000,20,768]
    # np.save(test_extract_npy, verctor)
    # print(u'输出路径：%s.npy' % test_extract_npy)


    # 测试集B榜
    data = load_data(test_extract_json_b)#[[,,,,],]
    verctor = convert(data)#[1000,20,768]
    np.save(test_extract_npy_b, verctor)
    print(u'输出路径：%s.npy' % test_extract_npy_b)

