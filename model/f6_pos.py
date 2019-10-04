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

print("-")
for df in [train_transaction,test_transaction]:
    if df.shape[0] == train_transaction.shape[0]:
        card1_dict = {}
        card1_amt_sum_dict = {}
        card1_amt_dict = {}
    df['card1'] = df['card1'].fillna(-1)
    mean_amt_card1_np = []
    count_uamt_card1_np = []
    mean_t_card1_np = []
    count_card1_np = []
    amt_to_mean_amt_card1_np = []
    cache = df[['TransactionDT','TransactionAmt','card1','uid']].values
    for i in range(cache.shape[0]):
        t = cache[i,0]
        amt = cache[i,1]
        card1 = cache[i,2]
        uid = cache[i,3]
        if card1_dict.get(card1,-1) == -1:
            card1_dict[card1] = [t]
        else:
            card1_dict[card1].append(t)
        if card1_amt_dict.get(card1,-1) == -1:
            card1_amt_dict[card1] = {}
            card1_amt_dict[card1][amt] = 1
        else:
            card1_amt_dict[card1][amt] = card1_amt_dict[card1].get(amt,0) + 1
        card1_amt_sum_dict[card1] = card1_amt_sum_dict.get(card1,0) + amt
        
        if(len(card1_dict[card1]) < 10):
            mean_t_card1 = np.mean(list(map(lambda x:t-x,card1_dict[card1])))
        else:
            mean_t_card1 = np.mean(list(map(lambda x:t-x,card1_dict[card1][-10:])))
        mean_amt_card1 = (card1_amt_sum_dict[card1] - amt)/(len(card1_dict[card1]) - 1 + 0.001)
        amt_to_mean_amt_card1 = amt - mean_amt_card1
        count_uamt_card1 = ((card1_amt_dict[card1][amt]) - 1)/(len(card1_dict[card1]) - 1 + 0.001)
        count_card1 = len(card1_dict[card1])    
        
        mean_amt_card1_np.append(mean_amt_card1)
        count_uamt_card1_np.append(count_uamt_card1)
        mean_t_card1_np.append(mean_t_card1)
        count_card1_np.append(count_card1)
        amt_to_mean_amt_card1_np.append(amt_to_mean_amt_card1)
        if i % 100000 == 0:
            print(i)


    df['mean_amt_card1'] = np.array(mean_amt_card1_np)
    df['count_uamt_card1'] = np.array(count_uamt_card1_np)
    df['mean_t_card1'] = np.array(mean_t_card1_np)
    df['count_card1'] = np.array(count_card1_np)
    df['amt_to_mean_amt_card1'] = np.array(amt_to_mean_amt_card1_np)
    
# col = ['C' + str(x) for x in range(14)]    
# for df in [train_transaction,test_transaction]:
    
# train_res = train_transaction[['pos_card1','pos_card2','pos_amt','pos_uid','pos2_card1','pos2_card2','pos2_amt','pos2_uid',
#                            'pos_t_card1','pos_t_card2','pos_t_amt','pos_t_uid','pos2_t_card1','pos2_t_card2','pos2_t_amt','pos2_t_uid']] 
train_res = train_transaction[['mean_amt_card1','count_uamt_card1','mean_t_card1','amt_to_mean_amt_card1']]        
train_res.to_csv("../input/f6_train.csv") 
test_res = test_transaction[['mean_amt_card1','count_uamt_card1','mean_t_card1','amt_to_mean_amt_card1']]        
test_res.to_csv("../input/f6_test.csv") 

