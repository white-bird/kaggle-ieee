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

from scipy.optimize import curve_fit

train_f5 = pd.read_csv('../input/fd_train3.csv', index_col='TransactionID')
test_f5 = pd.read_csv('../input/fd_test3.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)

train_f5 = pd.read_csv('../input/fe_train2.csv', index_col='TransactionID')
test_f5 = pd.read_csv('../input/fe_test2.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)

train_f5 = pd.read_csv('../input/fi_train4.csv', index_col='TransactionID')
test_f5 = pd.read_csv('../input/fi_test4.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)

debug = True
if debug:
    train_f5 = pd.read_csv('../input/xgb_26_train_debug.csv', index_col='TransactionID')
    test_f5 = pd.read_csv('../input/xgb_26_test_debug.csv', index_col='TransactionID')
    train_transaction = train_transaction.merge(train_f5.append(test_f5), how='left', left_index=True, right_index=True)
#     train_transaction['pred23'] = train_transaction['pred_1'] + train_transaction['pred_2'] + train_transaction['pred_3']
#     del  train_transaction['pred_1'],train_transaction['pred_2'], train_transaction['pred_3']
#     test_transaction['pred23'] = 0.5
    
else:
    train_f5 = pd.read_csv('../input/xgb_26_train.csv', index_col='TransactionID')
    test_f5 = pd.read_csv('../input/xgb_26_test.csv', index_col='TransactionID')
    train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
    test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)  
#     train_transaction['pred23'] = train_transaction['pred_1'] + train_transaction['pred_2'] + train_transaction['pred_3']
#     del  train_transaction['pred_1'],train_transaction['pred_2'], train_transaction['pred_3']
#     test_transaction['pred23'] = test_transaction['pred_1'] + test_transaction['pred_2'] + test_transaction['pred_3']
#     del  test_transaction['pred_1'],test_transaction['pred_2'], test_transaction['pred_3'] 
    

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

keys = ['ukey','ukey2','tempkey','ukey3','card123456_add1_D15_series','card123456_add1_D2_series','card123456_add1_D11_series','card123456_add1_D8_series']
count = 0
cols = []
ukey_dict = {}  # ukey set(id)
ukey_dict2 = {}  # id pred23 
cache = train_transaction[keys + ['TransactionDT','isFraud']].values


import math
def func(x,a,b):
#     return 1/(1+math.e**(-(a*x+b)))
    return 1/(1+(a*x+b))

for j in range(len(keys)):
    print(keys[j])
    ukey_dict = {}
    ukey2_dict = {}
    x_np = []
    y_np = []
    true_np = []
    for i in range(cache.shape[0]):
        isFraud = float(cache[i,-1]) 
        ukey = cache[i,j]
        t = float(cache[i,-2]) 
        ukey_dict[ukey] = ukey_dict.get(ukey,[])
        for k in range(max(len(ukey_dict[ukey]) - 10,0),len(ukey_dict[ukey])):
            dt = (t - ukey_dict[ukey][k][1])//(3600)
            if ukey_dict[ukey][k][0] == 1:
                x_np.append(dt)
                y_np.append(isFraud)
                true_np.append(isFraud)
#             else:
#                 x_np.append(dt)
#                 y_np.append(1)
#                 true_np.append(isFraud)
        ukey_dict[ukey].append([isFraud,t])
        
    x_np = np.array(x_np)
    y_np = np.array(y_np)
    popt,pcov=curve_fit(func,x_np,y_np,[0,0])
    a,b = popt
    print(popt)
    error1 = []
    error2 = []
    for xi in range(x_np.shape[0]):
        if true_np[xi] == 0:
            error1.append(abs(func(x_np[xi],a,b) - y_np[xi]))  
        if true_np[xi] == 1:
            error2.append(abs(y_np[xi] - func(x_np[xi],a,b)))
    print(np.mean(error1),np.mean(error2),np.sum(error1),np.sum(error2),len(error1),len(error2))