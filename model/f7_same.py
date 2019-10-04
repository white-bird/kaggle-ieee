import os

import numpy as np
import pandas as pd
import random

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')

train_transaction['hour'] = train_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
test_transaction['hour'] = test_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
train_transaction['TransactionDT2'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24 * 7)))
test_transaction['TransactionDT2'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24 * 7)))

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)



train_test = train_transaction.append(test_transaction)
train_test['card1'] = train_test['card1'].fillna(-1)
train_test['card2'] = train_test['card2'].fillna(-1)
train_test['addr1'] = train_test['addr1'].fillna(-1)
train_test['P_emaildomain'] = train_test['P_emaildomain'].fillna(-1)
# train_test['TransactionDT'] = train_test['TransactionDT'].map(lambda x:(x//(3600*24*7)))



cache = train_test[['hour','TransactionDT','card1','card2','addr1','P_emaildomain','uid']].values
isFraud = train_test[['isFraud']].values

amt_uid = {}


f1 = open('../input/f7_deepwalk2.csv','w')
f = f1
index_feature = {}
index = 0
import pickle
# maxnums = 200000
# maxlength = 128
for i in range(cache.shape[0]):
    hour = (cache[i,0])//3600 
    amt = "amt_" + str(cache[i,1])
    card1 = "card1_" + str(int(cache[i,2]))
    card2 = "card2_" + str(int(cache[i,3]))
    addr1 = "addr1_" + str(int(cache[i,4]))
    P_emaildomain = "P_emaildomain_" + str(cache[i,5])
    uid = "uid_" + str(cache[i,6])
    amt_uid[amt] = amt_uid.get(amt,[])
    
    
    
    amt_uid[amt] = list(filter(lambda x:abs(x[0] - hour) < 7 * 24,amt_uid[amt]))
    uids = list(map(lambda x:x[1], amt_uid[amt]))
    amt_uid[amt].append((hour,uid))
    if len(uids) == 0:
        continue
    for uidx in uids + [uid]:
        if index_feature.get(uidx,-1) == -1:
            index_feature[uidx] = index
            index += 1  
#     if index_feature.get(amt,-1) == -1:
#         index_feature[amt] = index
#         index += 1
    for uidx in uids:    
        print("\t".join([str(index_feature[uid]),str(index_feature[uidx])]),file=f)
    
    

f1.close()
print(len(index_feature))
with open('emb2/index_feature2.pkl', 'wb') as f:
    pickle.dump(index_feature, f)  

import os
os.system("python proNE.py -graph ../input/f7_deepwalk2.csv -emb1 emb2/sparse2.emb -emb2 emb2/spectral2.emb -dimension 8 -step 5 -theta 0.5 -mu 0.2")
