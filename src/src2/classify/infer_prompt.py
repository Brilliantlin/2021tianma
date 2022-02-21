# 数据筛选
from openprompt.plms import load_plm
import pandas as pd
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import json
import gc
from pandarallel import pandarallel
from LAC import LAC
from sklearn.metrics.pairwise import cosine_similarity,pairwise_distances
from tqdm.auto import tqdm
lac = LAC(mode='lac')
text = u"LAC是个优秀的分词工具"
lac_result = lac.run(text)
print(lac_result)
pandarallel.initialize(nb_workers = 5)
gc.collect()

def infer(valid_dataloader, promptModel):
    promptModel.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(tqdm(valid_dataloader, disable=True)):
        inputs = inputs.cuda()
        logits = promptModel(inputs)
        logits = F.softmax(logits,-1)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(logits.cpu().tolist())
    return allpreds

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
model_path = "../../FinBERT_L-12_H-768_A-12_pytorch/"
test = pd.read_csv('../datasets/Test_B.csv')
#读取实体统一库
with open('../datasets/entity_vocab.txt','r' ) as f:
    entitys = f.readlines()
    entitys = [x.strip() for x in entitys]
test['Text_new'] = test['Text'].parallel_apply(lambda x: entityResolution(x[:600]))
test['Label'] = 0

valid_set = []
for i, line in test.iterrows():
    line = line.to_dict()
    valid_set.append(
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
classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "金融科技新闻", "其他新闻"
]
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
promptModel = promptModel.cuda()
valid_dataloader = PromptDataLoader(dataset=valid_set,
                                    template=promptTemplate,
                                    tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass,
                                    max_seq_length=512,
                                    decoder_max_length=3,
                                    batch_size=24,
                                    shuffle=False,
                                    teacher_forcing=False,
                                    predict_eos_token=False,
                                    truncate_method="head")


allpreds = []
for seed in  tqdm([235,5192,905,7813,2895]):
    path = 'model_seed%s.bin' % (seed)
    data = torch.load(path)
    promptModel.load_state_dict(data)
    preds = infer(valid_dataloader,promptModel)
    allpreds.append(preds)
res = np.array(allpreds)
res = np.array([np.array(i)[:,1] for i in allpreds])
np.save('../results/prompt_probs.npy',res)
res = np.array([np.where(i>0.3,1,0) for i in res])

test['Label'] = np.where(res.sum(axis=0) > 3,1,0)
test['Abstract'] = ''

test[['ID','Label','Abstract']].to_excel('../results/final_prompt.xls',index=None,encoding='utf-8')