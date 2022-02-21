import os

import numpy as np
import sklearn.model_selection
from bert4keras.backend import keras, search_layer, K,set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm.auto import tqdm
from keras.layers import Dropout, Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import f1_score

from mytools.utils.myfile import makedirs

set_gelu('tanh')  # 切换gelu版本

maxlen = 512
batch_size = 8
# bert配置
# config_path = '/wangjin_fix_1/pre_model/tf/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/wangjin_fix_1/pre_model/tf/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/wangjin_fix_1/pre_model/tf/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

config_path = '/wangjin_fix_1/lin/2021BDCI/天马2021/FinBERT_L-12_H-768_A-12_tf/bert_config.json'
checkpoint_path = '/wangjin_fix_1/lin/2021BDCI/天马2021/FinBERT_L-12_H-768_A-12_tf/bert_model.ckpt'
dict_path = '/wangjin_fix_1/lin/2021BDCI/天马2021/FinBERT_L-12_H-768_A-12_tf/vocab.txt'

le = LabelEncoder()
def load_data(filename,mode='train',seed = 2012):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    df = pd.read_csv(filename)
    # df['Text'] = df['Domain'] + df['Text']

    if mode == 'train':
        train_pos = df[df.Label == 1]
        train_neg = df[df.Label == 0]
        train_neg = train_neg.sample(2000,random_state=seed,replace = False)
        df = pd.concat([train_pos,train_neg]) #train
        data = []
        for i,line in tqdm(df.iterrows(),total = df.shape[0]):
            data.append((line['Text'],line['Label']))
    else:
        df['Label'] = [1] * 212 + [0] * 1288
        data = []
        for i, line in tqdm(df.iterrows(), total=df.shape[0]):
            data.append((line['Text'],line['Label']))
    return data


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text,maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def build_model():
    # 加载预训练模型
    bert = build_transformer_model(
        model='bert',
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        return_keras_model=False,
    )

    output = Dropout(rate=0.5)(bert.model.output)
    output = Dense(
        units=2, activation='softmax', kernel_initializer=bert.initializer
    )(output)
    model = keras.models.Model(bert.model.input, output)
    # model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5,),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        metrics=['accuracy'],
    )
    return model


def evaluate(data):
    y = []
    y_hat = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        y.extend(y_true)
        y_hat.extend(y_pred)
    print(y_hat)
    fscore = f1_score(y,y_hat,pos_label=1,average='binary')
    return fscore

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.
        self.best_epoch = 0

    def on_batch_end(self, batch, logs=None):
        pass
        # if batch % 5 == 0:
        #     val_acc = evaluate(test_generator)
        #     if val_acc > self.best_val_acc:
        #         self.best_val_acc = val_acc
        #         self.best_epoch = batch
        #         if not os.path.exists('weights'):
        #             os.mkdir('weights')
        #         model.save_weights('weights/best_model_seed_%s_step_%s.weights' % (seed,batch))
        #     print(
        #         u'batch:%s val_fscore: %.5f, best_fscore: %.5f' %
        #         (batch,val_acc, self.best_val_acc)
        #     )

def get_random_list(seed):
    np.random.seed(seed)
    random_seeds = np.random.randint(0, 10000, size=5)
    print('Gnerate random seeds:', random_seeds)
    return random_seeds
def makedirs(prefix):
    '''prefix：文件夹目录，可以递归生成'''
    if not os.path.exists(prefix):
        os.makedirs(prefix)
if __name__ == '__main__':


    # 转换数据集
    # 加载数据集
    SEED = 2021
    seeds = get_random_list(SEED)
    makedirs('weights_adv_%s' % (SEED))
    for seed in seeds:
        train_data = load_data('../datasets/Train.csv',mode='train',seed=seed)
        test_data = load_data('../datasets/Test_B.csv',mode='test')

        train_generator = data_generator(train_data, batch_size)
        test_generator = data_generator(test_data, 8)

        model = build_model()
        evaluator = Evaluator()
        adversarial_training(model, 'Embedding-Token', 0.5)
        for i in range(4):
            model.fit(
                    test_generator.forfit(),
                    steps_per_epoch=len(test_generator),
                    epochs=1,
                    callbacks=[evaluator],
                )

            print('score%s' % (evaluate(test_generator)))

        # kf = StratifiedKFold()
        # fold = 1
        # for train_index, valid_index in kf.split(train_data,[x[1] for x in train_data]):
        #     tra = [train_data[i] for i in train_index]
        #     val = [train_data[i] for i in valid_index]
        #
        #     train_generator = data_generator(tra, batch_size)
        #     valid_generator = data_generator(val, batch_size)
        #     test_generator = data_generator(test_data, batch_size)
        #
        #     evaluator = Evaluator()
        #     model = build_model()
        #     model.fit(
        #         train_generator.forfit(),
        #         steps_per_epoch=len(train_generator),
        #         epochs=20,
        #         callbacks=[evaluator]
        #     )
        #     fold += 1
