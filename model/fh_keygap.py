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

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

train_f5 = pd.read_csv('../input/fd_train3.csv', index_col='TransactionID')
test_f5 = pd.read_csv('../input/fd_test3.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)

train_f5 = pd.read_csv('../input/fe_train2.csv', index_col='TransactionID')
test_f5 = pd.read_csv('../input/fe_test2.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)


# debug = True
# if debug:
#     train_f5 = pd.read_csv('../input/xgb_26_train_debug.csv', index_col='TransactionID')
#     test_f5 = pd.read_csv('../input/xgb_26_test_debug.csv', index_col='TransactionID')
#     train_transaction = train_transaction.merge(train_f5.append(test_f5), how='left', left_index=True, right_index=True)
# #     train_transaction['pred23'] = train_transaction['pred_1'] + train_transaction['pred_2'] + train_transaction['pred_3']
# #     del  train_transaction['pred_1'],train_transaction['pred_2'], train_transaction['pred_3']
# #     test_transaction['pred23'] = 0.5
    
# else:
#     train_f5 = pd.read_csv('../input/xgb_26_train.csv', index_col='TransactionID')
#     test_f5 = pd.read_csv('../input/xgb_26_test.csv', index_col='TransactionID')
#     train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
#     test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)  
# #     train_transaction['pred23'] = train_transaction['pred_1'] + train_transaction['pred_2'] + train_transaction['pred_3']
# #     del  train_transaction['pred_1'],train_transaction['pred_2'], train_transaction['pred_3']
# #     test_transaction['pred23'] = test_transaction['pred_1'] + test_transaction['pred_2'] + test_transaction['pred_3']
# #     del  test_transaction['pred_1'],test_transaction['pred_2'], test_transaction['pred_3'] 
    

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)

train_transaction['uid2'] = train_transaction['uid'].astype(str)+'_'+train_transaction['addr1'].astype(str)+'_'+train_transaction['P_emaildomain'].astype(str)
test_transaction['uid2'] = test_transaction['uid'].astype(str)+'_'+test_transaction['addr1'].astype(str)+'_'+test_transaction['P_emaildomain'].astype(str)

train_transaction['TransactionDT2'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600*24*7)))
test_transaction['TransactionDT2'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600*24*7)))

train_transaction['TransactionDT3'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600*24)))
test_transaction['TransactionDT3'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600*24)))

train_transaction['ukey3'] = train_transaction['TransactionDT2'].astype(str) + train_transaction['uid2'].astype(str) + train_transaction['TransactionAmt'].astype(str)
test_transaction['ukey3'] = test_transaction['TransactionDT2'].astype(str) + test_transaction['uid2'].astype(str) + test_transaction['TransactionAmt'].astype(str)

train_transaction['ukey2'] = train_transaction['TransactionDT3'].astype(str) + train_transaction['uid2'].astype(str) 
test_transaction['ukey2'] = test_transaction['TransactionDT3'].astype(str) + test_transaction['uid2'].astype(str) 


# if debug:
#     split_pos = train_transaction.shape[0]*4//5
#     test_transaction = train_transaction.iloc[split_pos:,:]
#     train_transaction = train_transaction.iloc[:split_pos,:]

keys = ['ukey','ukey2','tempkey','ukey3']
count = 0
cols = []

for j in range(len(keys)):
    
    key = keys[j]
    ukey_dict = {}  # ukey set(id)
    ukey_dict2 = {}  # id pred23 
    cache = train_transaction[[key] + ['TransactionDT','uid','uid2']].values

    np1 = []
    np2 = []
    np3 = []
    np4 = []
    

    for i in range(cache.shape[0]):
        ukey = cache[i,0]
        TransactionDT = float(cache[i,1]) 
        uid = cache[i,2]
        uid2 = cache[i,3]
        ukey_dict[uid] = ukey_dict.get(uid,[])
        ukey_dict2[uid2] = ukey_dict2.get(uid2,[])
        r1 = ""
        r2 = ""
        r3 = ""
        r4 = ""
        if len(ukey_dict[uid]) >= 1:
            for j in range(min(15,len(ukey_dict[uid]))):
                pos = -j - 1
                if ukey_dict[uid][pos][1] == ukey:
                    r1 = j
                    r2 = TransactionDT - ukey_dict[uid][pos][0]
                    break
        if len(ukey_dict2[uid2]) >= 1:
            for j in range(min(15,len(ukey_dict2[uid2]))):
                pos = -j - 1
                if ukey_dict2[uid2][pos][1] == ukey:
                    r3 = j
                    r4 = TransactionDT - ukey_dict2[uid2][pos][0]
                    break
        ukey_dict[uid].append([TransactionDT,ukey])
        ukey_dict2[uid2].append([TransactionDT,ukey])
        np1.append(r1)
        np2.append(r2)
        np3.append(r3)
        np4.append(r4)
        
    train_transaction[key + "_uidcount"] = np.array(np1)
    train_transaction[key + "_uiddt"] = np.array(np2)
    train_transaction[key + "_uid2count"] = np.array(np3)
    train_transaction[key + "_uid2dt"] = np.array(np4)
    cols.extend([key + "_uidcount",key + "_uiddt",key + "_uid2count"])

for j in range(len(keys)):
    
    key = keys[j]
    ukey_dict = {}  # ukey set(id)
    ukey_dict2 = {}  # id pred23 
    cache = test_transaction[[key] + ['TransactionDT','uid','uid2']].values

    np1 = []
    np2 = []
    np3 = []
    np4 = []
    

    for i in range(cache.shape[0]):
        ukey = cache[i,0]
        TransactionDT = float(cache[i,1]) 
        uid = cache[i,2]
        uid2 = cache[i,3]
        ukey_dict[uid] = ukey_dict.get(uid,[])
        ukey_dict2[uid2] = ukey_dict2.get(uid2,[])
        r1 = ""
        r2 = ""
        r3 = ""
        r4 = ""
        if len(ukey_dict[uid]) >= 1:
            for j in range(min(15,len(ukey_dict[uid]))):
                pos = -j - 1
                if ukey_dict[uid][pos][1] == ukey:
                    r1 = j
                    r2 = TransactionDT - ukey_dict[uid][pos][0]
                    break
        if len(ukey_dict2[uid2]) >= 1:
            for j in range(min(15,len(ukey_dict2[uid2]))):
                pos = -j - 1
                if ukey_dict2[uid2][pos][1] == ukey:
                    r3 = j
                    r4 = TransactionDT - ukey_dict2[uid2][pos][0]
                    break
        ukey_dict[uid].append([TransactionDT,ukey])
        ukey_dict2[uid2].append([TransactionDT,ukey])
        np1.append(r1)
        np2.append(r2)
        np3.append(r3)
        np4.append(r4)
        
    test_transaction[key + "_uidcount"] = np.array(np1)
    test_transaction[key + "_uiddt"] = np.array(np2)
    test_transaction[key + "_uid2count"] = np.array(np3)
    test_transaction[key + "_uid2dt"] = np.array(np4) 
    
train_transaction[cols].to_csv('../input/fh_train.csv',header = True)
test_transaction[cols].to_csv('../input/fh_test.csv',header = True)
