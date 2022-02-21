import pandas as pd
from tqdm.auto import tqdm
import json

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


# train = pd.read_csv('../datasets/Train.csv')
# test = pd.read_csv('../datasets/Test_A.csv')
# train.columns = ['ID','Domain','text','label','summary']
# test.columns = ['ID','Domain','text']

test_b = pd.read_csv('../datasets/Test_B.csv')
test_b.columns = ['text','ID']

def convert_to_extract(train,file,mode = 'train'):
    data = []
    max_sen_num = 132
    if mode == 'train':
        for i,line in tqdm(train.iterrows(),total = train.shape[0]):
            line = line.to_dict()
            splits = mysplit_keep_puntuation(line['text'],punctuation='。')[:max_sen_num]

            select_index = []
            for i,l in enumerate(splits):
                if l in line['summary']:
                    select_index.append(str(i))
            d = tuple([splits,select_index,line['summary'],line['label']])
            data.append(d)
    else:
         for i,line in tqdm(train.iterrows(),total = train.shape[0]):
            line = line.to_dict()
            splits = mysplit_keep_puntuation(line['text'],punctuation='。')[:max_sen_num]
            d = tuple([splits,[],'',0])
            data.append(d)
    with open(file, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    return data
# data_ds = convert_to_extract(train,'../datasets/train_extract.json',mode='train')
# test_ds = convert_to_extract(test,'../datasets/test_extract_B.json',mode='test')

test_ds = convert_to_extract(test_b,'../datasets/test_extract_B.json',mode='test')