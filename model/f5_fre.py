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
LENGTH = 1

for df in [train_transaction,test_transaction]:
    if True:
        card1_dict = {}
        card2_dict = {}
        card3_dict = {}
        card1_fre_dict = {}
        card2_fre_dict = {}
        card3_fre_dict = {}
    df['card1'] = df['card1'].fillna(-1)
    df['card2'] = df['card2'].fillna(-1)
    df['card3'] = df['card3'].fillna(-1)
    card1_fre_np = []
    card2_fre_np = []
    card3_fre_np = []
    
    cache = df[['TransactionDT','TransactionAmt','card1','card2','uid']].values
    for i in range(cache.shape[0]):
        t = cache[i,0]
        card1 = cache[i,2]
        card2 = cache[i,3]
        card3 = cache[i,4]
        card1_fre = -1
#         card1_fre2 = -1
        if card1_dict.get(card1,-1) == -1:
            card1_fre_dict[card1] = []
            card1_dict[card1] = t
        else:
            if card1 != -1:
                card1_fre = t - card1_dict[card1]
                card1_fre_dict[card1].append(card1_fre)
#                 if len(card1_fre_dict[card1]) > 2:
#                     card1_fre = np.mean(card1_fre_dict[card1][-LENGTH:])
        card1_dict[card1] = t
        card2_fre = -1
#         card2_fre2 = -1
        if card2_dict.get(card2,-1) == -1:
            card2_fre_dict[card2] = []
            card2_dict[card2] = t
        else:
            if card2 != -1:
                card2_fre = t - card2_dict[card2]
                card2_fre_dict[card2].append(card2_fre)
#                 if len(card2_fre_dict[card2]) > 2:
#                     card2_fre = np.mean(card2_fre_dict[card2][-LENGTH:])
        card2_dict[card2] = t
        card3_fre = -1
#         card3_fre2 = -1
        if card3_dict.get(card3,-1) == -1:
            card3_fre_dict[card3] = []
            card3_dict[card3] = t
        else:
            if card3 != -1:
                card3_fre = t - card3_dict[card3]
                card3_fre_dict[card3].append(card3_fre)
#                 if len(card3_fre_dict[card3]) > 2:
#                     card3_fre = np.mean(card3_fre_dict[card3][-LENGTH:])
#                 card3_fre2 = np.std(card3_fre_dict[card3][-LENGTH:])
        card3_dict[card3] = t        
        card1_fre_np.append(card1_fre)
        card2_fre_np.append(card2_fre)
        card3_fre_np.append(card3_fre)
        
        
    df['fre_card1_1'] = np.array(card1_fre_np)
    df['fre_card2_1'] = np.array(card2_fre_np)
    df['fre_uid_1'] = np.array(card3_fre_np)
    
for df in [test_transaction,train_transaction]:
    if True:
        card1_dict = {}
        card2_dict = {}
        card3_dict = {}
        card1_fre_dict = {}
        card2_fre_dict = {}
        card3_fre_dict = {}
    df['card1'] = df['card1'].fillna(-1)
    df['card2'] = df['card2'].fillna(-1)
    df['card3'] = df['card3'].fillna(-1)
    card1_fre_np = []
    card2_fre_np = []
    card3_fre_np = []
    cache = df[['TransactionDT','TransactionAmt','card1','card2','uid']].values
    for i in range(cache.shape[0]-1,-1,-1):
        t = cache[i,0]
        card1 = cache[i,2]
        card2 = cache[i,3]
        card3 = cache[i,4]
        card1_fre = -1
#         card1_fre2 = -1
        if card1_dict.get(card1,-1) == -1:
            card1_fre_dict[card1] = []
            card1_dict[card1] = t
        else:
            if card1 != -1:
                card1_fre = card1_dict[card1] - t
                card1_fre_dict[card1].append(card1_fre)
#                 if len(card1_fre_dict[card1]) > 2:
#                     card1_fre = np.mean(card1_fre_dict[card1][-LENGTH:])
        card1_dict[card1] = t
        card2_fre = -1
#         card2_fre2 = -1
        if card2_dict.get(card2,-1) == -1:
            card2_fre_dict[card2] = []
            card2_dict[card2] = t
        else:
            if card2 != -1:
                card2_fre = card2_dict[card2] - t
                card2_fre_dict[card2].append(card2_fre)
#                 if len(card2_fre_dict[card2]) > 2:
#                     card2_fre = np.mean(card2_fre_dict[card2][-LENGTH:])
        card2_dict[card2] = t
        card3_fre = -1
#         card3_fre2 = -1
        if card3_dict.get(card3,-1) == -1:
            card3_fre_dict[card3] = []
            card3_dict[card3] = t 
        else:
            if card3 != -1:
                card3_fre = card3_dict[card3] - t
                card3_fre_dict[card3].append(card3_fre)
#                 if len(card3_fre_dict[card3]) > 2:
#                     card3_fre = np.mean(card3_fre_dict[card3][-LENGTH:])
#                 card3_fre2 = np.std(card3_fre_dict[card3][-LENGTH:])
        card3_dict[card3] = t        
        card1_fre_np.append(card1_fre)
        card2_fre_np.append(card2_fre)
        card3_fre_np.append(card3_fre)
        
    df['fre_card1_2'] = np.flip(np.array(card1_fre_np),axis = 0)
    df['fre_card2_2'] = np.flip(np.array(card2_fre_np),axis = 0)
    df['fre_uid_2'] = np.flip(np.array(card3_fre_np),axis = 0)
    
train_res = train_transaction[['fre_uid_1','fre_uid_2']]        
train_res.to_csv("../input/f5_train.csv") 
test_res = test_transaction[['fre_uid_1','fre_uid_2']]        
test_res.to_csv("../input/f5_test.csv") 

# from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=8)
# LENGTH = 8
# train_test = train_transaction.append(test_transaction)
# if True:
#     df = train_test
#     card1_fre_dict = {}
#     card2_fre_dict = {}
#     amt_fre_dict = {}
#     df['card1'] = df['card1'].fillna(-1)
#     df['card2'] = df['card2'].fillna(-1)
#     df['card3'] = df['card3'].fillna(-1)
#     card1_fre_np = []
#     card2_fre_np = []
#     amt_fre1_np = []
#     cache = df[['TransactionDT','TransactionAmt','card1','card2','card3']].values
#     for i in range(cache.shape[0]):
#         t = cache[i,0]
#         amt = cache[i,1]
#         card1 = cache[i,2]
#         card2 = cache[i,3]
#         card3 = cache[i,4]
#         amt_fre = [0] * LENGTH
#         if amt_fre_dict.get(amt,-1) == -1:
#             amt_fre_dict[amt] = []
#             amt_fre_dict[amt].append(t)
#         else:
#             if amt != -1:
#                 amt_fre = ([0] * LENGTH + list(map(lambda x:t - x,amt_fre_dict[amt][-LENGTH:])))[-LENGTH:]
#                 amt_fre_dict[amt].append(t)
#         card1_fre = [0] * LENGTH
#         if card1_fre_dict.get(card1,-1) == -1:
#             card1_fre_dict[card1] = []
#             card1_fre_dict[card1].append(t)
#         else:
#             if card1 != -1:
#                 card1_fre = ([0] * LENGTH + list(map(lambda x:t - x,card1_fre_dict[card1][-LENGTH:])))[-LENGTH:]
#                 card1_fre_dict[card1].append(t)
#         card2_fre = [0] * LENGTH
#         if card2_fre_dict.get(card2,-1) == -1:
#             card2_fre_dict[card2] = []
#             card2_fre_dict[card2].append(t)
#         else:
#             if card2 != -1:
#                 card2_fre = ([0] * LENGTH + list(map(lambda x:t - x,card2_fre_dict[card2][-LENGTH:])))[-LENGTH:]
#                 card2_fre_dict[card2].append(t)
                
#         card1_fre_np.append(card1_fre)
#         card2_fre_np.append(card2_fre)
#         amt_fre1_np.append(amt_fre)
# #         print(card1_fre)
# #         print(card2_fre)
# #         print(amt_fre)

#     card1_fre2_np = []
#     card2_fre2_np = []
#     amt_fre2_np = []
#     card1_fre_dict = {}
#     card2_fre_dict = {}
#     amt_fre_dict = {}
#     cache = np.flip(df[['TransactionDT','TransactionAmt','card1','card2','card3']].values,axis = 0)
#     for i in range(cache.shape[0]):
#         t = cache[i,0]
#         amt = cache[i,1]
#         card1 = cache[i,2]
#         card2 = cache[i,3]
#         card3 = cache[i,4]
#         amt_fre = [0] * LENGTH
#         if amt_fre_dict.get(amt,-1) == -1:
#             amt_fre_dict[amt] = []
#             amt_fre_dict[amt].append(t)
#         else:
#             if amt != -1:
#                 amt_fre = ([0] * LENGTH + list(map(lambda x:t - x,amt_fre_dict[amt][-LENGTH:])))[-LENGTH:]
#                 amt_fre_dict[amt].append(t)
#         card1_fre = [0] * LENGTH
#         if card1_fre_dict.get(card1,-1) == -1:
#             card1_fre_dict[card1] = []
#             card1_fre_dict[card1].append(t)
#         else:
#             if card1 != -1:
#                 card1_fre = ([0] * LENGTH + list(map(lambda x:t - x,card1_fre_dict[card1][-LENGTH:])))[-LENGTH:]
#                 card1_fre_dict[card1].append(t)
#         card2_fre = [0] * LENGTH
#         if card2_fre_dict.get(card2,-1) == -1:
#             card2_fre_dict[card2] = []
#             card2_fre_dict[card2].append(t)
#         else:
#             if card2 != -1:
#                 card2_fre = ([0] * LENGTH + list(map(lambda x:t - x,card2_fre_dict[card2][-LENGTH:])))[-LENGTH:]
#                 card2_fre_dict[card2].append(t)
                
#         card1_fre2_np.append(card1_fre)
#         card2_fre2_np.append(card2_fre)
#         amt_fre2_np.append(amt_fre)
        
#     card1_fre_np = np.concatenate([np.array(card1_fre_np),np.flip(np.array(card1_fre2_np),axis = 0)],axis = 1)
#     card2_fre_np = np.concatenate([np.array(card2_fre_np),np.flip(np.array(card2_fre2_np),axis = 0)],axis = 1)
#     amt_fre_np = np.abs(np.concatenate([np.array(amt_fre1_np),np.flip(np.array(amt_fre2_np),axis = 0)],axis = 1))
    
#     amt_mean = []
#     amt_std = []
#     amt_sum = []
#     amt_len1 = []
#     amt_len2 = []
#     n1 = np.array(amt_fre1_np)
#     n2 = np.abs(np.flip(np.array(amt_fre2_np),axis = 0))
#     for i in range(amt_fre_np.shape[0]):
#         l1 = list(filter(lambda x:x!=0,n1[i,:]))
#         l2 = list(filter(lambda x:x!=0,n2[i,:]))
#         amt_mean.append(np.mean(l1+l2))
#         amt_std.append(np.std(l1+l2))
#         amt_sum.append(np.sum(l1+l2))
#         l1 = list(filter(lambda x:x!=0 and x < 24 * 3600 * 7,n1[i,:]))
#         l2 = list(filter(lambda x:x!=0 and x < 24 * 3600 * 7,n2[i,:]))
#         amt_len1.append(len(l1))
#         amt_len2.append(len(l2))
                       
        
#     df['amt_dur_mean'] = np.array(amt_mean)
#     df['amt_dur_std'] = np.array(amt_std)
#     df['amt_sum'] = np.array(amt_sum)
#     df['amt_len1'] = np.array(amt_len1)    
#     df['amt_len2'] = np.array(amt_len2)   
    
#     target = np.concatenate([card1_fre_np,card2_fre_np,amt_fre_np],axis = 1)
#     target = np.log(np.abs(target) + 1)
#     svd.fit(target)
#     res = svd.transform(target)
#     for i in range(8):
#         df['svd_'+str(i)] = res[:,i]

# train_res = train_test[['svd_'+str(i) for i in range(8)] + ['amt_dur_mean','amt_dur_std','amt_sum','amt_len1','amt_len2']].iloc[:train_transaction.shape[0],:]      
# train_res.to_csv("../input/f5_train2.csv") 
# test_res = train_test[['svd_'+str(i) for i in range(8)]+ ['amt_dur_mean','amt_dur_std','amt_sum','amt_len1','amt_len2']].iloc[train_transaction.shape[0]:,:]        
# test_res.to_csv("../input/f5_test2.csv") 