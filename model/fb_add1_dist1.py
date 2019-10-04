import os

import numpy as np
import pandas as pd
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)




train_transaction['uid3'] = train_transaction['uid'].astype(str)+'_'+train_transaction['P_emaildomain'].astype(str)
test_transaction['uid3'] = test_transaction['uid'].astype(str)+'_'+test_transaction['P_emaildomain'].astype(str)

for col in [[["uid3"],["addr1"]]]:
    for col2 in col[1]:
        
        train_transaction['next'+"_".join(col[0])+col2+"_"] = train_transaction.sort_values(col[0] + ['TransactionDT'])[col2].shift(-1)
        train_transaction['next'+"_".join(col[0])+col2+"_"] = train_transaction['next'+"_".join(col[0])+col2+"_"].astype(str) + "_" +train_transaction[col2].astype(str)
        test_transaction['next'+"_".join(col[0])+col2+"_"] = test_transaction.sort_values(col[0] + ['TransactionDT'])[col2].shift(-1)
        test_transaction['next'+"_".join(col[0])+col2+"_"] = test_transaction['next'+"_".join(col[0])+col2+"_"].astype(str) + "_" +test_transaction[col2].astype(str)
        
        train_transaction['before'+"_".join(col[0])+col2+"_"] = train_transaction.sort_values(col[0] + ['TransactionDT'])[col2].shift(1)
        train_transaction['before'+"_".join(col[0])+col2+"_"] = train_transaction['before'+"_".join(col[0])+col2+"_"].astype(str) + "_" +train_transaction[col2].astype(str)
        test_transaction['before'+"_".join(col[0])+col2+"_"] = test_transaction.sort_values(col[0] + ['TransactionDT'])[col2].shift(1)
        test_transaction['before'+"_".join(col[0])+col2+"_"] = test_transaction['before'+"_".join(col[0])+col2+"_"].astype(str) + "_" +test_transaction[col2].astype(str)
  
      

train_test = train_transaction.append(test_transaction)
cache = train_test[['TransactionDT','nextuid3addr1_','beforeuid3addr1_']].values       
f1 = open('../input/fb_deepwalk.csv','w')
f = f1
index_feature = {}
index = 0
import pickle
# maxnums = 200000
# maxlength = 128
for i in range(cache.shape[0]):
    
    addr1arr = cache[i,1].split("_")
    if index_feature.get(addr1arr[0],-1) == -1:
        index_feature[addr1arr[0]] = index
        index += 1  
    if index_feature.get(addr1arr[1],-1) == -1:
        index_feature[addr1arr[1]] = index
        index += 1  
        
    print("\t".join([str(index_feature[addr1arr[0]]),str(index_feature[addr1arr[1]])]),file=f)
    
    addr1arr = cache[i,2].split("_")
    if index_feature.get(addr1arr[0],-1) == -1:
        index_feature[addr1arr[0]] = index
        index += 1  
    if index_feature.get(addr1arr[1],-1) == -1:
        index_feature[addr1arr[1]] = index
        index += 1  
        
    print("\t".join([str(index_feature[addr1arr[0]]),str(index_feature[addr1arr[1]])]),file=f)
    
f1.close()


train_res = train_transaction[['nextuid3addr1_','beforeuid3addr1_']]        
train_res.to_csv("../input/fb_train.csv") 
test_res = test_transaction[['nextuid3addr1_','beforeuid3addr1_']]        
test_res.to_csv("../input/fb_test.csv") 

print(len(index_feature))
with open('emb3/index_feature.pkl', 'wb') as f:
    pickle.dump(index_feature, f)   


import os
os.system("python proNE.py -graph ../input/fb_deepwalk.csv -emb1 emb3/sparse.emb -emb2 emb3/spectral.emb -dimension 8 -step 5 -theta 0.5 -mu 0.2")
    