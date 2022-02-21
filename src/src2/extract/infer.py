import pandas as pd
import numpy as np
import extract_vectorize as convert
import extract_model as extract
from snippets import *
from tqdm.auto import tqdm

if __name__ == '__main__':
    vector = np.load('../datasets/test_extract_B.npy')
    model = extract.build_model()
    data = convert.load_data('../datasets/test_extract_B.json')
    train = pd.read_csv('../datasets/Train.csv')

    threshold = 0.92
    preds = []
    for fold in tqdm(range(15)):
        model.load_weights('./weights/extract_model.%s.weights' % fold)
        pred = model.predict(vector)[:, :, 0]
        pred = np.where(pred > threshold, 1, 0)
        preds.append(pred)
    preds = np.array(preds)
    select = preds.sum(axis=0) >= 7
    print(select.sum())

    res = []
    for ind, (line, sel) in enumerate(zip(data, select)):
        summary = ''
        for i, flag in zip(line, sel):
            if flag:
                summary += i
            else:
                break
        res.append(summary)
    test = pd.read_csv('../datasets/Test_B.csv')
    test['Abstract'] = res
    test['Label'] = 0

    test[['ID', 'Label', 'Abstract']].to_excel('../results/final_extract.xls', index=None, encoding='utf-8')










