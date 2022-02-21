import json

import pandas as pd
import sklearn.metrics

from main import *
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.metrics import roc_auc_score
maxlen = 512
# def auc(y_true, y_pred):
#     auc = roc_auc_score(y_true,y_pred)
#     return auc

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

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, example in self.sample(random):
            text = example['Text']
            label = example['class']
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

if __name__ == '__main__':
    SEED = 2021
    makedirs('select_sample_weights')

    # 对训练集进行5折划分
    data = pd.read_csv('../datasets/Train.csv')
    target_domain = pd.read_csv('../datasets/Test_B.csv')
    data['class'] = 0
    target_domain['Label'] = 1
    target_domain['class'] = 1
    pos = data[data.Label == 1]
    neg = data[data.Label == 0].sample(5000, random_state=SEED).reset_index(drop=True)
    kf = KFold(5,random_state=SEED,shuffle=True)

    train_index_pos = []
    valid_index_pos = []
    for train_index,valid_index in kf.split(pos):
        train_index_pos.append(train_index)
        valid_index_pos.append(valid_index)
    train_index_neg = []
    valid_index_neg = []
    for train_index,valid_index in kf.split(neg):
        train_index_neg.append(train_index)
        valid_index_neg.append(valid_index)

    preds_data = []
    for i in range(5):
        print('fold %s traing' % (i))
        train_data = pd.concat([pos.iloc[train_index_pos[i]],neg.iloc[train_index_neg[i]]]).sample(frac=1)
        train_data = train_data[['Text','class','Label']]
        train_data = pd.concat([train_data,target_domain])
        train_data = train_data.to_dict('records')
        train_generator = data_generator(train_data,batch_size=24)

        valid_data = pd.concat([pos.iloc[valid_index_pos[i]],neg.iloc[valid_index_neg[i]]]).sample(frac=1)
        valid_data = valid_data[['Text','class','Label']]
        valid_data = valid_data.to_dict('records')
        valid_generator = data_generator(valid_data,batch_size=32)

        model  = build_model()
        model.fit(
                train_generator.forfit(),
                steps_per_epoch=len(train_generator),
                epochs=1,
            )
        model.save_weights(r'select_sample_weights/fold_%s.weights' % (i))

        print('fold %s predicting' % (i))
        #预测验证集
        pred = []
        for x_,y_ in valid_generator:
            pred_tmp = model.predict(x_)[:, 1]
            pred.extend(pred_tmp)
        pred = np.array(pred)

        for record,p in zip(valid_data,pred):
            record['pred'] = p
            preds_data.append(record)
    new = pd.DataFrame(preds_data)
    new.to_csv('../datasets/select.csv',index=None)



