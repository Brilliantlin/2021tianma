#!/usr/bin/env python
# coding: utf-8

# ## utils

# In[1]:


import _pickle as pickle
import numpy as np
import jieba
import tensorflow as tf
import keras
from sklearn import metrics
import collections
from tqdm.auto import tqdm

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def word2idx_forward(doc, vocab, MAX_SENT_LENGTH, MAX_SENTS):
    sents = []
    for sent in doc:
        sent = [vocab[word] for word in jieba.lcut(sent) if word in vocab]
        sents.append(sent[:MAX_SENT_LENGTH] + [0] * (MAX_SENT_LENGTH - len(sent)))
    sents = np.array(sents[:MAX_SENTS])
    if len(sents) == MAX_SENTS:
        return sents
    else:
        return np.r_[sents, np.array([[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(sents)))]


def word2idx_backward(doc, vocab, MAX_SENT_LENGTH, MAX_SENTS):
    sents = []
    for sent in doc[::-1]:
        sent = [vocab[word] for word in jieba.lcut(sent) if word in vocab]
        sents.append(sent[:MAX_SENT_LENGTH] + [0] * (MAX_SENT_LENGTH - len(sent)))
    sents = np.array(sents[:MAX_SENTS])
    if len(sents) == MAX_SENTS:
        return sents[::-1]
    else:
        return np.r_[sents[::-1], np.array([[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(sents)))]

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    inputs: [bs,senctences,sentence_len]
    """
    if length is None:
        length = max([len(x) for x in inputs]) 
    pad_width = [(0, 0) for _ in np.shape(inputs[0])] 
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


class data_generator_forward(object):
    """数据生成器
    """
    def __init__(self, data, batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS, train = True,buffer_size = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.vocab = vocab
        self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
        self.MAX_SENTS = MAX_SENTS
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size # 向下取整
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000
    def __len__(self):
        return self.steps
    
    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield chaches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                        while caches:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]
            data = generator()
        else:
            data = iter(self.data)
        
        d_current = next(data) 
        for d_next in data:
            yield False, d_current
            d_current = d_next
        
        yield True, d_current
        
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if self.train == True:
            for is_end, (ids, text, label) in self.sample(random):
                batch_token_ids.append(word2idx_forward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield batch_token_ids, batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for is_end, (ids, text, label) in self.sample(random):
                batch_token_ids.append(word2idx_forward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    yield batch_token_ids
                    batch_token_ids = []
        
    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d
    
    def forpredict(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d

class data_generator_backward(object):
    """数据生成器
    """
    def __init__(self, data, batch_size,vocab, MAX_SENT_LENGTH, MAX_SENTS, train = True,buffer_size = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.vocab = vocab
        self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
        self.MAX_SENTS = MAX_SENTS
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000
    def __len__(self):
        return self.steps
    
    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield chaches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                        while caches:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]
            data = generator()
        else:
            data = iter(self.data)
        
        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next
        
        yield True, d_current
        
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if self.train == True:
            for is_end, (ids, label,text) in self.sample(random):
                batch_token_ids.append(word2idx_backward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                batch_labels.append([label])
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield batch_token_ids, batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        else:
            for is_end, (ids, text, label) in self.sample(random):
                batch_token_ids.append(word2idx_backward(text, self.vocab, self.MAX_SENT_LENGTH, self.MAX_SENTS))
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    yield batch_token_ids
                    batch_token_ids = []
        
    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d
    
    def forpredict(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# cut words
def cut_text(sentence):
    tokens = lac.run(sentence)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


# # 实体统一

# In[2]:



import pandas as pd
import os
import pandas as pd
from tqdm.auto import tqdm
import gc
from pandarallel import pandarallel
pandarallel.initialize(5)
gc.collect()

from LAC import LAC
lac = LAC(mode='lac')
lac.add_word('【科技实体】')
text = u"LAC是个优秀的分词工具"
lac_result = lac.run(text)
print(lac_result)


train = pd.read_csv('datasets/Train.csv')
test = pd.read_csv('datasets/Test_B.csv')


with open('lac/entity_vocab.txt','r' ) as f:
    entitys = f.readlines()
    entitys = [x.strip() for x in entitys]

import json
def entityResolution(line):
    line = lac.run(line)
    words,postag = line[0],line[1]
    new_str = ''
    for w in words:
        if w in entitys:
            new_str += '【科技实体】'
        else:
            new_str += w
    return new_str


train['Text_new'] = train['Text'].parallel_apply(lambda x: entityResolution(x[:512])) 
test['Text_new'] = test['Text'].parallel_apply(lambda x: entityResolution(x[:512]))


# In[3]:


def reposition(data):
    mid = data['text']
    data.drop(labels=['text'], axis=1,inplace = True)
    data.insert(1, 'text', mid)
    return data
train.drop(columns=['Domain','Abstract','Text'],inplace=True)
train.rename(columns={'ID':'id','Text_new':'text','Label':'label'},inplace=True)
train = reposition(train)

test.drop(columns=['Text'],inplace=True)
test.rename(columns={'ID':'id','Text_new':'text'},inplace=True)
test = reposition(test)


# ## pretrainW2V

# In[4]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import jieba
import gensim
from keras.preprocessing.text import Tokenizer

train = train
test = test

train_labels = train.label.values

stop_words = []

vocab_set_path = 'model_han_fin/vocab_set.pkl'
test['label'] = 0

train['text'] = train['text'].parallel_apply(lambda x: cut_text(x)[0])
test['text'] = test['text'].parallel_apply(lambda x: cut_text(x)[0])
#词语计数

word_cnt = collections.Counter()
for line in tqdm(np.r_[train['text'].values,test['text'].values],desc= 'word count ing'):
    word_cnt.update(line)
min_count = 50
word_cnt = [k for k in tqdm(word_cnt,desc='filt ing...') if word_cnt[k] > min_count] 
word_cnt = {k:v for v,k in enumerate(word_cnt)}

vocab = word_cnt

train['text'] = train['text'].parallel_apply(lambda x:[i for i in x if i in vocab])
test['text'] = test['text'].parallel_apply(lambda x:[i for i in x if i in vocab])

# save vocab
save_variable(vocab,vocab_set_path) 

print('开始训练词向量')
vector_size = 100 #2
model = gensim.models.Word2Vec(size=vector_size, window=5, min_count=50, workers=5, sg=0, iter=8, seed=2021)
model.build_vocab(np.r_[train['text'].values,test['text'].values])
model.train(np.r_[train['text'].values,test['text'].values], total_examples=model.corpus_count, epochs=model.iter)
model.save("model_han_fin/w2v.model")
word2vec_save = 'model_han_fin/word2vec_model.txt'
model.wv.save_word2vec_format(word2vec_save, binary=False)


# ## model

# In[5]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__( **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='att_weight',
                                shape=(input_shape[1], input_shape[1]),
                                initializer='uniform',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

import tensorflow.keras.backend as K

class SetLearningRate:
    """
    	layer learning rate
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # learning rate
        self.is_ada = is_ada # if adam

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embed', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb 
                else:
                    lamb = self.lamb**0.5 
                K.set_value(weight, K.eval(weight) / lamb) 
                setattr(self.layer, key, weight * lamb) 
        return self.layer(inputs)


class my_HAN_model(object):
    def __init__(self, MAX_SENT_LENGTH, MAX_SENTS, embedding_matrix, vocab):
    	self.MAX_SENT_LENGTH = MAX_SENT_LENGTH
    	self.MAX_SENTS = MAX_SENTS
    	self.embedding_matrix = embedding_matrix
    	self.vocab = vocab
    def create_model(self):
        sentence_inputs = Input(shape=(self.MAX_SENT_LENGTH,), dtype='float64')
        
        embed = Embedding(len(self.vocab) + 1, 100, input_length=self.MAX_SENT_LENGTH, weights=[self.embedding_matrix], trainable=True)
        embed = SetLearningRate(embed, 0.001, True)(sentence_inputs)
        l_lstm = Bidirectional(GRU(128, return_sequences=True))(embed)
        l_dense = TimeDistributed(Dense(64))(l_lstm)
        print(l_dense.shape)
        l_att = AttentionLayer()(l_dense)
        print(l_att.shape)
        sentEncoder = Model(sentence_inputs, l_att)
        
        review_input = Input(shape=(self.MAX_SENTS, self.MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(GRU(128, return_sequences=True))(review_encoder)
        l_dense_sent = TimeDistributed(Dense(64))(l_lstm_sent)
        l_att_sent = AttentionLayer()(l_dense_sent)
        outputs = Dense(1, activation='sigmoid')(l_att_sent)
        
        self.model = Model(review_input, outputs=outputs)
        self.compile()
        return self.model
    
    def compile(self):
        self.model.compile(
             loss='binary_crossentropy',
            optimizer=Adam(1e-3),  
            metrics=['accuracy'],
        )

epsilon = 1e-5
smooth = 1
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


# ## train

# In[6]:


from pandarallel import pandarallel
pandarallel.initialize(nb_workers = 4)
def mysplit_keep_puntuation(s, punctuation='!~。？?,，；'):
    res_list = []
    buff = ''

    for c in s:
        buff += c
        if c in punctuation:
            res_list.append(buff)
            buff = ''
    if buff != '':
        res_list.append(buff)
    return res_list
def reposition(data):
    mid = data['text']
    data.drop(labels=['text'], axis=1,inplace = True)
    data.insert(1, 'text', mid)
    return data
def dataProcess(data,vocab):
    data['text'] = data['text'].parallel_apply(lambda x: cut_text(x)[0])
    data['text'] = data['text'].parallel_apply(lambda x:''.join([i for i in x if i in vocab]))
    data['text'] = data['text'].parallel_apply(lambda x:mysplit_keep_puntuation(x,'。'))
    return data


# In[7]:


import os
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from sklearn import metrics
import jieba
import gensim
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
resample_flag = True
MAX_SENT_LENGTH = 100
MAX_SENTS = 10
EMBEDDING_DIM = 100

vocab = load_variavle('model_han_fin/vocab_set.pkl')
w2v_model = gensim.models.Word2Vec.load("model_han_fin/w2v.model")

train = pd.read_csv("datasets/Train.csv")
train['Text_new'] = train['Text'].parallel_apply(lambda x: entityResolution(x[:512]))
train.drop(columns=['Domain','Abstract','Text'],inplace=True)
train.rename(columns={'ID':'id','Text_new':'text','Label':'label'},inplace=True)
train = reposition(train)


# In[8]:


def evaluate(data,model):
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_preds = np.r_[y_preds, model.predict(x_true)[:,0]]
        y_trues = np.r_[y_trues, y_true[:, 0]]
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    return  metrics.auc(fpr, tpr)

def evaluate_recall(data,model):
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_preds = np.r_[y_preds, model.predict(x_true)[:,0]]
        y_trues = np.r_[y_trues, y_true[:, 0]]
    fscore = metrics.f1_score(y_trues, y_preds > 0.5, pos_label=1, average='binary')
    
    print(fscore)
    thresholds = list(np.arange(0.0, 1.0, 0.001))
    acc_scores = np.zeros(shape=(len(thresholds)))
    for index, elem in tqdm(enumerate(thresholds),total=len(thresholds),disable = True):
        y_pred_prob = (y_preds > elem).astype('int')
        acc_scores[index] =  metrics.f1_score(y_trues,y_pred_prob,pos_label=1,average='binary')
    index = np.argmax(acc_scores)
    thresholdOpt = round(thresholds[index], ndigits = 5)
    best = round(acc_scores[index], ndigits = 5)
    return  best,thresholdOpt
class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_name,valid_generator,model):
        self.best_val_acc = 0
        self.model_name = model_name
        self.valid_generator = valid_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator, self.model) #auc
        best,threshold = evaluate_recall(self.valid_generator, self.model)
        if (best > self.best_val_acc):
            print('1')
            self.best_val_acc = best
            self.best_threshold = threshold
            self.model.save_weights(self.model_name)
        print(
            u'val_auc: %.5f, best_val_auc: %.5f, val_f1: %.5f' %
            (val_acc, self.best_val_acc, best)
        )
        logs['val_auc'] = val_acc
        logs['val_recall'] = best


# In[9]:



embedding_matrix = np.zeros((len(vocab) + 1, 100))
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

# 5折
train_entity = train[train['text'].str.contains('【科技实体】')]
train_pos = train[train['label']==1]
train_neg = train[train['label']==0].sample(2000,random_state=123)
train_df = pd.concat([train_entity,train_pos,train_neg]).drop_duplicates()
train_df = dataProcess(train_df,vocab)

train_batch_size = 8 

K.clear_session()
md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
model = md.create_model()
train_generator = data_generator_forward(train_df.values, train_batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS) 
print("开始训练")
model.fit_generator(
    train_generator.forfit(), 
    steps_per_epoch=len(train_generator),
    epochs= 2,

)
model.save_weights('model_han_fin/best_model{}.weights'.format(1))


# ## predict

# In[10]:


test = pd.read_csv('datasets/Test_B.csv')
test['Text_new'] = test['Text'].parallel_apply(lambda x: entityResolution(x[:512]))
test.drop(columns =['Text'],inplace=True)
test.rename(columns={'ID':'id','Text_new':'text'},inplace=True)
test = reposition(test)
test = dataProcess(test,vocab)


# In[11]:


'''
	Use thie file to generate result.
'''
import numpy as np

import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from bert4keras.backend import keras, set_gelu
from keras.layers import *
from keras import backend as K
from sklearn import metrics
import gensim
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

batch_size = 32
test['Label'] = 0


K.clear_session()
md = my_HAN_model(MAX_SENT_LENGTH = MAX_SENT_LENGTH, MAX_SENTS=MAX_SENTS, embedding_matrix=embedding_matrix,vocab=vocab)
model = md.create_model()
test_generator = data_generator_forward(test.values, batch_size, vocab, MAX_SENT_LENGTH, MAX_SENTS, train=False)
model.load_weights(r'model_han_fin/best_model{}.weights'.format(1))
probs = model.predict_generator(test_generator.forpredict(),
    steps=len(test_generator))[:,0]
    


# # 结果保存

# In[23]:


y_pred = np.array(probs)
np.save('result/han_probs.npy',y_pred)
r = []
for pred in y_pred:
    pred = np.where(pred > 0.4,1,0)
    r.append(pred)
r = np.array(r)
test = pd.read_csv('datasets/Test_B.csv')
submit_data = test[['ID']]
submit_data["Label"]=r
submit_data.to_csv("result/result_han_fin.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




