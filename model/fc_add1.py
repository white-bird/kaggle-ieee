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



from collections import Counter
import math
def entropy(pr):
    total = sum(list(map(lambda x:x[1],pr.items())))
    pr = list(map(lambda x:x[1],pr.items()))
    log2 = math.log2
    ent = 0
    for i in pr:
        p = float(i) / total
        ent += (-p) * log2(p)
    return ent
  
from math import sqrt

def confidence(ups, n):
    if n == 0:
        return 0
    z = 1.0 #1.44 = 85%, 1.96 = 95%
    phat = float(ups) / n
    return ((phat + z*z/(2*n) - z * sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n))      

debug_pos = 0.8  
fre1 = 10000  # 10000
fre = 30000 # 50000
train_test = train_transaction.append(test_transaction)
train_test['addr1'] = train_test['addr1'].fillna(-1)
train_test['P_emaildomain'] = train_test['P_emaildomain'].fillna(-1)
train_test['R_emaildomain'] = train_test['R_emaildomain'].fillna(-1)
cache = train_test[['TransactionDT','addr1','uid','isFraud']].values    
index = train_test.index
uid_addr1_dict = {}
uid_addr1_temp_dict = {}
uid_addr1_fraud_dict = {}
uid_addr1_fraud_temp_dict = {}

uid_addr1_fraud_np = []
uid_addr1_fraud2_np = []
pos_np = []
for i in range(cache.shape[0]):
    addr1 = cache[i,1]
    uid = cache[i,2]
    isFraud = cache[i,3]
    
    d = uid_addr1_dict.get(uid,{})
    d2 = uid_addr1_fraud_dict.get(uid,{})
#     uid_addr1_en_np.append(entropy(d))
#     uid_addr1_per_np.append(d.get(addr1,0)/(sum(list(map(lambda x:x[1],d.items())))+0.01))
    if d.get(addr1,0) == 0:
        uid_addr1_fraud_np.append(-1)
    else:
        uid_addr1_fraud_np.append(d2.get(addr1,0)/(d.get(addr1,0)+0.01))
#         uid_addr1_fraud_np.append(confidence(d2.get(addr1,0),(d.get(addr1,0)+0.01)))
    uid_addr1_fraud2_np.append(d2.get(addr1,0))                       
                           
    if index[i] == 3567757:
        print(addr1,uid,d2.get(addr1,0),d.get(addr1,0))
    
    if i <= int(train_transaction.shape[0] * debug_pos):
        pos_np.append(i%fre)
    else:
        pos_np.append(min(fre,i - int(train_transaction.shape[0] * debug_pos)))
    
    if isFraud == 1:
        uid_addr1_fraud_temp_dict[uid] = uid_addr1_fraud_temp_dict.get(uid,{})
        uid_addr1_fraud_temp_dict[uid][addr1] = uid_addr1_fraud_temp_dict[uid].get(addr1,0) + 1

    uid_addr1_temp_dict[uid] = uid_addr1_temp_dict.get(uid,{})
    uid_addr1_temp_dict[uid][addr1] = uid_addr1_temp_dict[uid].get(addr1,0) + 1
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_fraud_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_fraud_dict[k] = uid_addr1_fraud_dict.get(k,{})
                uid_addr1_fraud_dict[k][k2] = uid_addr1_fraud_dict[k].get(k2,0) + v2
        uid_addr1_fraud_temp_dict = {}
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_dict[k] = uid_addr1_dict.get(k,{})
                uid_addr1_dict[k][k2] = uid_addr1_dict[k].get(k2,0) + v2    
        uid_addr1_temp_dict = {}

uid_addr1_dict = {}
uid_addr1_temp_dict = {}
uid_addr1_en_np = []
uid_addr1_per_np = []
for i in range(cache.shape[0]):
    addr1 = cache[i,1]
    uid = cache[i,2]
    isFraud = cache[i,3]
    
    d = uid_addr1_dict.get(uid,{})
    d2 = uid_addr1_fraud_dict.get(uid,{})
    uid_addr1_en_np.append(entropy(d))
    uid_addr1_per_np.append(d.get(addr1,0)/(sum(list(map(lambda x:x[1],d.items())))+0.01))
                           
                           

    uid_addr1_temp_dict[uid] = uid_addr1_temp_dict.get(uid,{})
    uid_addr1_temp_dict[uid][addr1] = uid_addr1_temp_dict[uid].get(addr1,0) + 1
    if (i % 5000 == 0 or i == train_transaction.shape[0]):
        for k,v in uid_addr1_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_dict[k] = uid_addr1_dict.get(k,{})
                uid_addr1_dict[k][k2] = uid_addr1_dict[k].get(k2,0) + v2     
        uid_addr1_temp_dict = {}
        
train_transaction['pos_np'] = np.array(pos_np)[:train_transaction.shape[0]]        
test_transaction['pos_np'] = np.array(pos_np)[train_transaction.shape[0]:]                 
train_transaction['uid_addr1_en'] = np.array(uid_addr1_en_np)[:train_transaction.shape[0]]       
train_transaction['uid_addr1_per'] = np.array(uid_addr1_per_np)[:train_transaction.shape[0]]    
train_transaction['uid_addr1_fraud'] = np.array(uid_addr1_fraud_np)[:train_transaction.shape[0]]   
train_transaction['uid_addr1_fraud2'] = np.array(uid_addr1_fraud2_np)[:train_transaction.shape[0]] 
test_transaction['uid_addr1_en'] = np.array(uid_addr1_en_np)[train_transaction.shape[0]:]       
test_transaction['uid_addr1_per'] = np.array(uid_addr1_per_np)[train_transaction.shape[0]:]    
test_transaction['uid_addr1_fraud'] = np.array(uid_addr1_fraud_np)[train_transaction.shape[0]:]    
test_transaction['uid_addr1_fraud2'] = np.array(uid_addr1_fraud2_np)[train_transaction.shape[0]:] 


cache = train_test[['TransactionDT','P_emaildomain','uid','isFraud']].values       
uid_addr1_dict = {}
uid_addr1_temp_dict = {}
uid_addr1_fraud_dict = {}
uid_addr1_fraud_temp_dict = {}

uid_P_emaildomain_fraud_np = []
uid_P_emaildomain_fraud2_np = []
for i in range(cache.shape[0]):
    addr1 = cache[i,1]
    uid = cache[i,2]
    isFraud = cache[i,3]
    
    d = uid_addr1_dict.get(uid,{})
    d2 = uid_addr1_fraud_dict.get(uid,{})
    if d.get(addr1,0) == 0:
        uid_P_emaildomain_fraud_np.append(-1)
    else:
        uid_P_emaildomain_fraud_np.append(d2.get(addr1,0)/(d.get(addr1,0)+0.01))
#         uid_P_emaildomain_fraud_np.append(confidence(d2.get(addr1,0),(d.get(addr1,0)+0.01)))
    uid_P_emaildomain_fraud2_np.append(d2.get(addr1,0))                        
                           
    
    if isFraud == 1:
        uid_addr1_fraud_temp_dict[uid] = uid_addr1_fraud_temp_dict.get(uid,{})
        uid_addr1_fraud_temp_dict[uid][addr1] = uid_addr1_fraud_temp_dict[uid].get(addr1,0) + 1

    uid_addr1_temp_dict[uid] = uid_addr1_temp_dict.get(uid,{})
    uid_addr1_temp_dict[uid][addr1] = uid_addr1_temp_dict[uid].get(addr1,0) + 1
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_fraud_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_fraud_dict[k] = uid_addr1_fraud_dict.get(k,{})
                uid_addr1_fraud_dict[k][k2] = uid_addr1_fraud_dict[k].get(k2,0) + v2
        uid_addr1_fraud_temp_dict = {}
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_dict[k] = uid_addr1_dict.get(k,{})
                uid_addr1_dict[k][k2] = uid_addr1_dict[k].get(k2,0) + v2    
        uid_addr1_temp_dict = {}

train_transaction['uid_P_emaildomain_fraud'] = np.array(uid_P_emaildomain_fraud_np)[:train_transaction.shape[0]]        
test_transaction['uid_P_emaildomain_fraud'] = np.array(uid_P_emaildomain_fraud_np)[train_transaction.shape[0]:] 
train_transaction['uid_P_emaildomain_fraud2'] = np.array(uid_P_emaildomain_fraud2_np)[:train_transaction.shape[0]]        
test_transaction['uid_P_emaildomain_fraud2'] = np.array(uid_P_emaildomain_fraud2_np)[train_transaction.shape[0]:] 

cache = train_test[['TransactionDT','R_emaildomain','uid','isFraud']].values       
uid_addr1_dict = {}
uid_addr1_temp_dict = {}
uid_addr1_fraud_dict = {}
uid_addr1_fraud_temp_dict = {}

uid_R_emaildomain_fraud_np = []
uid_R_emaildomain_fraud2_np = []
for i in range(cache.shape[0]):
    addr1 = cache[i,1]
    uid = cache[i,2]
    isFraud = cache[i,3]
    
    d = uid_addr1_dict.get(uid,{})
    d2 = uid_addr1_fraud_dict.get(uid,{})
    if d.get(addr1,0) == 0:
        uid_R_emaildomain_fraud_np.append(-1)
    else:
        uid_R_emaildomain_fraud_np.append(d2.get(addr1,0)/(d.get(addr1,0)+0.01))
#         uid_R_emaildomain_fraud_np.append(confidence(d2.get(addr1,0),(d.get(addr1,0)+0.01)))
    uid_R_emaildomain_fraud2_np.append(d2.get(addr1,0))                       
                           
    
    if isFraud == 1:
        uid_addr1_fraud_temp_dict[uid] = uid_addr1_fraud_temp_dict.get(uid,{})
        uid_addr1_fraud_temp_dict[uid][addr1] = uid_addr1_fraud_temp_dict[uid].get(addr1,0) + 1

    uid_addr1_temp_dict[uid] = uid_addr1_temp_dict.get(uid,{})
    uid_addr1_temp_dict[uid][addr1] = uid_addr1_temp_dict[uid].get(addr1,0) + 1
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_fraud_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_fraud_dict[k] = uid_addr1_fraud_dict.get(k,{})
                uid_addr1_fraud_dict[k][k2] = uid_addr1_fraud_dict[k].get(k2,0) + v2
        uid_addr1_fraud_temp_dict = {}
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_dict[k] = uid_addr1_dict.get(k,{})
                uid_addr1_dict[k][k2] = uid_addr1_dict[k].get(k2,0) + v2    
        uid_addr1_temp_dict = {}

train_transaction['uid_R_emaildomain_fraud'] = np.array(uid_R_emaildomain_fraud_np)[:train_transaction.shape[0]]        
test_transaction['uid_R_emaildomain_fraud'] = np.array(uid_R_emaildomain_fraud_np)[train_transaction.shape[0]:] 
train_transaction['uid_R_emaildomain_fraud2'] = np.array(uid_R_emaildomain_fraud2_np)[:train_transaction.shape[0]]        
test_transaction['uid_R_emaildomain_fraud2'] = np.array(uid_R_emaildomain_fraud2_np)[train_transaction.shape[0]:] 

cache = train_test[['TransactionDT','TransactionAmt','uid','isFraud']].values       
uid_addr1_dict = {}
uid_addr1_temp_dict = {}
uid_addr1_fraud_dict = {}
uid_addr1_fraud_temp_dict = {}

uid_TransactionAmt_fraud_np = []
uid_TransactionAmt_fraud2_np = []
for i in range(cache.shape[0]):
    addr1 = cache[i,1]
    uid = cache[i,2]
    isFraud = cache[i,3]
    
    d = uid_addr1_dict.get(uid,{})
    d2 = uid_addr1_fraud_dict.get(uid,{})
    if d.get(addr1,0) == 0:
        uid_TransactionAmt_fraud_np.append(-1)
    else:
        uid_TransactionAmt_fraud_np.append(d2.get(addr1,0)/(d.get(addr1,0)+0.01))
#         uid_TransactionAmt_fraud_np.append(confidence(d2.get(addr1,0),(d.get(addr1,0)+0.01)))
    uid_TransactionAmt_fraud2_np.append(d2.get(addr1,0))                      
                           
    
    if isFraud == 1:
        uid_addr1_fraud_temp_dict[uid] = uid_addr1_fraud_temp_dict.get(uid,{})
        uid_addr1_fraud_temp_dict[uid][addr1] = uid_addr1_fraud_temp_dict[uid].get(addr1,0) + 1

    uid_addr1_temp_dict[uid] = uid_addr1_temp_dict.get(uid,{})
    uid_addr1_temp_dict[uid][addr1] = uid_addr1_temp_dict[uid].get(addr1,0) + 1
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_fraud_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_fraud_dict[k] = uid_addr1_fraud_dict.get(k,{})
                uid_addr1_fraud_dict[k][k2] = uid_addr1_fraud_dict[k].get(k2,0) + v2
        uid_addr1_fraud_temp_dict = {}
    if (i % fre == 0 or i == int(train_transaction.shape[0] * debug_pos)) and i <= int(train_transaction.shape[0] * debug_pos):
        for k,v in uid_addr1_temp_dict.items():
            for k2,v2 in v.items():
                uid_addr1_dict[k] = uid_addr1_dict.get(k,{})
                uid_addr1_dict[k][k2] = uid_addr1_dict[k].get(k2,0) + v2    
        uid_addr1_temp_dict = {}

train_transaction['uid_TransactionAmt_fraud'] = np.array(uid_TransactionAmt_fraud_np)[:train_transaction.shape[0]]        
test_transaction['uid_TransactionAmt_fraud'] = np.array(uid_TransactionAmt_fraud_np)[train_transaction.shape[0]:] 
train_transaction['uid_TransactionAmt_fraud2'] = np.array(uid_TransactionAmt_fraud2_np)[:train_transaction.shape[0]]        
test_transaction['uid_TransactionAmt_fraud2'] = np.array(uid_TransactionAmt_fraud2_np)[train_transaction.shape[0]:] 

train_res = train_transaction[['uid_addr1_en','uid_addr1_per','uid_addr1_fraud','uid_P_emaildomain_fraud','uid_R_emaildomain_fraud','uid_TransactionAmt_fraud'
                            ,'uid_addr1_fraud2','uid_P_emaildomain_fraud2','uid_R_emaildomain_fraud2','uid_TransactionAmt_fraud2','pos_np']]        
train_res.to_csv("../input/fc_train.csv") 
test_res = test_transaction[['uid_addr1_en','uid_addr1_per','uid_addr1_fraud','uid_P_emaildomain_fraud','uid_R_emaildomain_fraud','uid_TransactionAmt_fraud','uid_addr1_fraud2','uid_P_emaildomain_fraud2','uid_R_emaildomain_fraud2','uid_TransactionAmt_fraud2','pos_np']]        
test_res.to_csv("../input/fc_test.csv") 

    