import numpy as np
import pandas as pd

extract = pd.read_excel('src2/results/final_extract.xls')
probs1 = np.load('src2/results/prompt_probs.npy')
probs2 = np.load('src1/result/han_probs.npy')
probs2 = probs2.reshape(-1, 1500)
probs = np.concatenate([probs1, probs2])

res = np.array([np.where(i > 0.520, 1, 0) for i in probs])
extract['Label'] = np.where(res.sum(axis = 0) >= 4, 1, 0)

extract.to_excel('results.xls',index=None)

print(1)