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

train_transaction['hour'] = train_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
test_transaction['hour'] = test_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
train_transaction['TransactionDT2'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24 * 7)))
test_transaction['TransactionDT2'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24 * 7)))

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)



train_test = train_transaction.append(test_transaction)
train_test['card1'] = train_test['card1'].fillna(-1)
# train_test['TransactionDT'] = train_test['TransactionDT'].map(lambda x:(x//(3600*24*7)))



cache = train_test[['TransactionDT','TransactionAmt','card1','card2','addr1','P_emaildomain','uid']].values
isFraud = train_test[['isFraud']].values

uid_amt = {}


count_uid_np = []
count_uid_2_np = []
count_uid_3_np = []
LENGTH = 24
LENGTH2 = 2
LENGTH3 = 12
for i in range(cache.shape[0]):
    hour = (cache[i,0])//3600 
    amt = cache[i,1]
    card1 = cache[i,2]
    uid = cache[i,6]
    uid_amt[uid] = uid_amt.get(uid,[])
    uid_amt[uid] = list(filter(lambda x:abs(x[0] - hour) < LENGTH,uid_amt[uid]))
    
    count_uid_np.append(len(list(map(lambda x:x[1], uid_amt[uid]))))
    count_uid_2_np.append(len(list(filter(lambda x:abs(x[0] - hour) < LENGTH2,uid_amt[uid]))))
    count_uid_3_np.append(len(list(filter(lambda x:abs(x[0] - hour) < LENGTH3,uid_amt[uid]))))
    
    uid_amt[uid].append((hour,amt))
    
uid_amt = {}
for i in range(cache.shape[0]-1,-1,-1):
    hour = (cache[i,0])//3600 
    amt = cache[i,1]
    card1 = cache[i,2]
    uid = cache[i,6]
    uid_amt[uid] = uid_amt.get(uid,[])
    uid_amt[uid] = list(filter(lambda x:abs(x[0] - hour) < LENGTH,uid_amt[uid]))
    
    count_uid_np[i]+=(len(list(map(lambda x:x[1], uid_amt[uid]))))
    count_uid_2_np[i]+=(len(list(filter(lambda x:abs(x[0] - hour) < LENGTH2,uid_amt[uid]))))
    count_uid_3_np[i]+=(len(list(filter(lambda x:abs(x[0] - hour) < LENGTH3,uid_amt[uid]))))
    
    uid_amt[uid].append((hour,amt))    
    
train_transaction['fre_l1_uid'] = np.array(count_uid_np)[:train_transaction.shape[0]]
test_transaction['fre_l1_uid'] = np.array(count_uid_np)[train_transaction.shape[0]:]
train_transaction['fre_l2_uid'] = np.array(count_uid_2_np)[:train_transaction.shape[0]]
test_transaction['fre_l2_uid'] = np.array(count_uid_2_np)[train_transaction.shape[0]:]
train_transaction['fre_l3_uid'] = np.array(count_uid_3_np)[:train_transaction.shape[0]]
test_transaction['fre_l3_uid'] = np.array(count_uid_3_np)[train_transaction.shape[0]:]

train_res = train_transaction[['fre_l1_uid','fre_l2_uid','fre_l3_uid']]        
train_res.to_csv("../input/f8_train.csv") 
test_res = test_transaction[['fre_l1_uid','fre_l2_uid','fre_l3_uid']]        
test_res.to_csv("../input/f8_test.csv") 