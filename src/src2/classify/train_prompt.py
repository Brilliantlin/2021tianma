#!/usr/bin/env python

import numpy as np
from paddlenlp import Taskflow 
import pandas as pd
import os
from tqdm.auto import tqdm
import gc 
import jieba
from pandarallel import pandarallel
from LAC import LAC
from sklearn.metrics.pairwise import cosine_similarity,pairwise_distances

lac = LAC(mode='lac')
text = u"LAC是个优秀的分词工具"
lac_result = lac.run(text)
print(lac_result)
pandarallel.initialize(5)
gc.collect()


# # utils

# In[2]:


def mysplit_keep_puntuation(s, punctuation='!~。？?,，；'):
    '''
    :param s: input string
    :param punctuation: 需要分割的标点
    :return: 切分结果，保留标点
    '''
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


# # read

# In[3]:


model_path = "../../FinBERT_L-12_H-768_A-12_pytorch/"


# In[4]:


train = pd.read_csv('../datasets/Train.csv')
test = pd.read_csv('../datasets/Test_B.csv')


# In[5]:


data_extract_npy = '../datasets/train_extract3.npy'
data_extract = np.load(data_extract_npy)

test_extract_npy = '../datasets/test_extract_B.npy'
test_extract = np.load(test_extract_npy)

print(data_extract.shape,test_extract.shape)

train_length = np.array([(i.sum(-1) != 0 ).sum() for i in data_extract])
test_length = np.array([(i.sum(-1) != 0 ).sum() for i in test_extract])

t = data_extract.sum(1)
div_mat = np.ones_like(t)
for i in range(len(div_mat)):
    div_mat[i] *= train_length[i]
train_vec = t / div_mat

t = test_extract.sum(1)
div_mat = np.ones_like(t)
for i in range(len(div_mat)):
    div_mat[i] *= test_length[i]
test_vec = t / div_mat


select = cosine_similarity(test_vec,train_vec)   #第一行的值是a1中的第一个行向量与a2中所有的行向量之间的余弦相似度

index = select.argsort()[:,-30:]

ind = []
for i in index:
    for j in i:
        ind.append(j)

ind = list(set(ind))

train_tmp = train.iloc[ind].reset_index(drop = True)
train_pos = train[train.Label==1]
train = pd.concat([train_tmp,train_pos]).drop_duplicates()





#读取实体统一库
with open('../datasets/entity_vocab.txt','r' ) as f:
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



train['Text_new'] = train['Text'].parallel_apply(lambda x: entityResolution(x[:600]))
test['Text_new'] = test['Text'].parallel_apply(lambda x: entityResolution(x[:600]))


# 数据筛选
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
# 
import numpy as np
from sklearn.metrics import f1_score

def eval(valid_dataloader, promptModel):
    promptModel.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(tqdm(valid_dataloader, disable=True)):
        inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return f1_score(alllabels, allpreds, pos_label=1, average='binary')


def get_random_list(seed):
    np.random.seed(seed)
    random_seeds = np.random.randint(0, 10000, size=5)
    print('Gnerate random seeds:', random_seeds)
    return random_seeds


seeds = get_random_list(1)
for seed in seeds:
    print('🐱' * 20 + 'run seed %s' % (seed) + '🐱' * 20)
    train_entity = train[train['Text_new'].str.contains('【科技实体】')]
    train_pos = train[train['Label'] == 1]
    train_neg = train[train['Label'] == 0].sample(1000,
                                                  random_state=seed,
                                                  replace=False)
    train_df = pd.concat([train_entity,
                          train_neg]).drop_duplicates()
    classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
        "金融科技新闻", "其他新闻"
    ]
    dataset = []
    valid_set = []
    for i, line in train_df.iterrows():
        line = line.to_dict()
        dataset.append(
            InputExample(
                guid=i,
                text_a=line['Text_new'],
                text_b='',
                label=int(line['Label']),
            ), )



    plm, tokenizer, model_config, WrapperClass = load_plm('bert', model_path)
    #模板构建
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} 他是 {"mask"}.',
        tokenizer=tokenizer,
    )
    #模型

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "金融科技新闻": ["是"],
            "其他新闻": ["不是"],
        },
        tokenizer=tokenizer,
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    train_dataloader = PromptDataLoader(dataset=dataset,
                                        template=promptTemplate,
                                        tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass,
                                        max_seq_length=512,
                                        decoder_max_length=3,
                                        batch_size=24,
                                        shuffle=True,
                                        teacher_forcing=False,
                                        predict_eos_token=False,
                                        truncate_method="head")

    best_score = 0
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
    ]
    promptModel = promptModel.cuda()
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    for epoch in range(2):
#         promptModel.train()
        tot_loss = 0 
        for step, inputs in enumerate(tqdm(train_dataloader)):
            inputs = inputs.cuda()
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
    data = promptModel.state_dict()
    torch.save(data, 'model_seed%s.bin' % (seed))
    del promptModel
    torch.cuda.empty_cache() 




