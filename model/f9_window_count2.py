import os

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)

train_transaction['uid'] = train_transaction['uid'].astype(str)+'_'+train_transaction['addr1'].astype(str)+'_'+train_transaction['addr2'].astype(str)
test_transaction['uid'] = test_transaction['uid'].astype(str)+'_'+test_transaction['addr1'].astype(str)+'_'+test_transaction['addr2'].astype(str)

LENGTH = 24
for df in [train_transaction,test_transaction]:
    window_count_uid_np = []
    window_ucount_uid_np = []
    uid_hour = {}
    cache = df[['TransactionDT','TransactionAmt','uid']].values
    for i in range(cache.shape[0]):
        amt = cache[i,1]
        uid = cache[i,2]
        hour = (cache[i,0])//3600
        
        uid_hour[uid] = uid_hour.get(uid,[])
        uid_hour[uid] = list(filter(lambda x:abs(x[0] - hour) < LENGTH,uid_hour[uid]))

        window_count_uid_np.append(len(list(map(lambda x:x[1], uid_hour[uid]))))
        window_ucount_uid_np.append(len(set(map(lambda x:x[1], uid_hour[uid]))))
    
        uid_hour[uid].append((hour,amt))

    df['window_count_uid_np'] = np.array(window_count_uid_np)
    df['window_ucount_uid_np'] = np.array(window_ucount_uid_np)
    
train_res = train_transaction[['window_count_uid_np','window_ucount_uid_np']]        
train_res.to_csv("../input/f9_train.csv") 
test_res = test_transaction[['window_count_uid_np','window_ucount_uid_np']]        
test_res.to_csv("../input/f9_test.csv") 

