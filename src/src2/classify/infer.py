import numpy as np
from tqdm.auto import tqdm
from main import build_model,load_data,data_generator
import  os

if __name__ == "__main__":

    path = 'weights_2021'
    model_list = os.listdir(path)
    model = build_model()

    test_data = load_data('../datasets/Test_A.csv', mode='test')
    test_loader = data_generator(test_data,32)

    preds = []
    for file in tqdm(model_list):
        model.load_weights(os.path.join(path,file))
        pred = []
        for x_,y_ in test_loader:
            pred_tmp = model.predict(x_)[:,1]
            pred.extend(pred_tmp)
        pred = np.array(pred)
        preds.append(pred)

    preds = np.array(preds)

    k = 280
    r = []
    thresholds = [0.3,0.2,0.3,0.55,0.55]
    for pred,th in zip(preds,thresholds):
        # topk = np.argsort(-pred)[:k]
        # pred = np.zeros(len(pred))
        # pred[topk] = 1
        pred = np.where(pred > th,1,0)
        print(pred.sum())
        r.append(pred)
    r = np.array(r)
    vote = r.sum(axis=0)
    res = (vote >= 3).astype(int)
    res.sum()

    with open('../results/class_base6.txt','w') as f:
        for i in res:
            f.write(str(i)+ '\n')
