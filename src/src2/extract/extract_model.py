#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 抽取式：主要模型
# 科学空间：https://kexue.fm

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import LayerNormalization
from bert4keras.optimizers import Adam
from bert4keras.snippets import open
from keras.layers import *
from keras.models import Model
from snippets import *
import numpy.ma as ma
# 配置信息
input_size = 768
hidden_size = 428
epochs = 20
batch_size = 64
threshold = 0.2
data_extract_json = data_json[:-5] + '_extract.json'
data_extract_npy = data_json[:-5] + '_extract3.npy'


fold = 0

# if len(sys.argv) == 1:
#     fold = 0
# else:
#     fold = int(sys.argv[1])


def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            l[1] = [int(x) for x in l[1]]
            D.append(l)
    return D


class ResidualGatedConv1D(Layer):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        super(ResidualGatedConv1D, self).build(input_shape)
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='same',
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name='alpha', shape=[1], initializer='zeros'
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = K.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        }
        base_config = super(ResidualGatedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model():
    x_in = Input(shape=(None, input_size))
    x = x_in

    x = Masking()(x)
    x = Dropout(0.1)(x)
    x = Dense(hidden_size, use_bias=False)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=2)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=4)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=8)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = ResidualGatedConv1D(hidden_size, 3, dilation_rate=1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(x_in, x)
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy']
    )
    # model.summary()
    return model

def evaluate(data, data_x, threshold=0.5):
    """验证集评估
    """
    y_pred = model.predict(data_x)[:, :, 0] # [9088,20]
    y_true = valid_y[:,:,0] #

    #掩盖pad
    mask = (data_x.sum(axis = -1) == 0)  # [num,20,768]
    y_pred = ma.masked_array(y_pred, mask=mask)
    y_true = ma.masked_array(y_true, mask=mask)
    print('参与计数句子数：%s'  % (mask.size - mask.sum()))

    thresholds = list(np.arange(0.0, 1.0, 0.001))
    acc_scores = np.zeros(shape=(len(thresholds)))
    for index, elem in tqdm(enumerate(thresholds),total=len(thresholds),disable = True ):
        # 修正概率
        y_pred_prob = (y_pred > elem).astype('int')
        # 计算f值
        acc_scores[index] = (y_true == y_pred_prob).mean()
    index = np.argmax(acc_scores)
    thresholdOpt = round(thresholds[index], ndigits = 5)
    best = round(acc_scores[index], ndigits = 5)

    y_pred = np.where(y_pred > threshold, 1, 0)
    return (y_true == y_pred).mean(),best,thresholdOpt

class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        acc,acc_t,threshold = evaluate(valid_data, valid_x, 0.5)
        if acc  >= self.best_metric:  # 保存最优
            self.best_metric = acc
            self.best_threshold = threshold
            model.save_weights('weights/extract_model.%s.weights' % fold)
        # print('[epoch %s ] [eval score: %s ] [最好阈值 %s  score: %s] [best score: %s] ' % (epoch,acc,threshold,acc_t,self.best_metric))

if __name__ == '__main__':
    print('.................fold %s...............'%(fold))
    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    data_y = np.zeros_like(data_x[..., :1])

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i, j] = 1

    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')

    train_x = data_split(data_x, fold, num_folds, 'train')
    valid_x = data_split(data_x, fold, num_folds, 'valid')

    train_y = data_split(data_y, fold, num_folds, 'train')
    valid_y = data_split(data_y, fold, num_folds, 'valid')

    # 启动训练
    evaluator = Evaluator()
    model =  build_model()
    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[evaluator]
    )
    print('🐱'*20,evaluator.best_threshold)

#[0.438,0.496, 0.522, 0.561,0.449,0.536,0.545,0.557,0.491,0.598,0.644, 0.506, 0.57,0.593, 0.49]