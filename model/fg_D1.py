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

old_col = [x for x in train_transaction.columns]
old_col2 = [x for x in test_transaction.columns]
change_col = [x for x in train_transaction.columns if x not in ['TransactionDT','isFraud']]

train_transaction['error'] = 1

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)


train_transaction['uid2'] = train_transaction['uid'].astype(str)+'_'+train_transaction['addr1'].astype(str)+'_'+train_transaction['P_emaildomain'].astype(str)
test_transaction['uid2'] = test_transaction['uid'].astype(str)+'_'+test_transaction['addr1'].astype(str)+'_'+test_transaction['P_emaildomain'].astype(str)

id_col = "D1,D4,D10,D15".split(",")
for col in id_col:
    train_transaction[col + "_day"] = train_transaction[col] - train_transaction['TransactionDT'].map(lambda x:(x//(3600*24)))
    test_transaction[col + "_day"] = test_transaction[col] - test_transaction['TransactionDT'].map(lambda x:(x//(3600*24)))
    
for col in id_col:    
    train_transaction[col] = train_transaction[col].fillna(-1)
    test_transaction[col] = test_transaction[col].fillna(-1)   
    train_transaction[col + "_day"] = train_transaction[col + "_day"].fillna(-1)
    test_transaction[col + "_day"] = test_transaction[col + "_day"].fillna(-1)  
    



cache = train_transaction[['uid','isFraud','ProductCD','TransactionAmt','addr1','TransactionDT'] + id_col + list(map(lambda x:x + "_day",id_col))].values
ukey_dict = {}
ukey2_dict = {}


count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    
    t = cache[i,5] // 3600
    D1_day = float(cache[i,6]) 
    D1 = float(cache[i,7]) 
    match_id = ["0"] * len(id_col)     
    match_col = []
    
    ukey = uid + "_" + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[train_transaction.index[i]] = train_transaction.index[i]
    if len(ukey_dict[ukey]) >= 1:
        for j in range(min(15,len(ukey_dict[ukey]))):
            match_id = ["0"] * len(id_col)     
            match_col = []
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 3 * 24:
                continue
                
            isfit = False
            for j in range(len(id_col)):
                if cache[i,6+j+ len(id_col)] != '-1' and cache[i,6+j] >= 5 and abs(cache[i,6+j+ len(id_col)] - ukey_dict[ukey][pos][3+j + len(id_col)]) <= 1:
                    isfit = True
                    if True:
                        match_col.append((id_col[j],cache[i,6+j]))
                        match_id[j] = cache[i,6+j]
#                 if cache[i,7+j] != 'nan' and ukey_dict[ukey][pos][5+j] != 'nan' and cache[i,7+j] != ukey_dict[ukey][pos][5+j]:
#                     isfit = False
#                     match_id = ["0"] * len(id_col) 
#                     break
#                     break
            if not isfit or len(list(filter(lambda x:x!="0",match_id)))<1:
                continue
                
            ukey2_dict[train_transaction.index[i]] = ukey2_dict[train_transaction.index[ukey_dict[ukey][pos][2]]]

                
            if ukey_dict[ukey][pos][0] + isFraud >= 1:
                count1 += 1
            if ukey_dict[ukey][pos][0] != isFraud:
                print(ukey,train_transaction.index[ukey_dict[ukey][pos][2]],train_transaction.index[i],ukey_dict[ukey][pos][-1],amt,ukey_dict[ukey][pos][0],isFraud,match_col)
                count2 += 1
            break
                    

    ukey_dict[ukey].append([isFraud,t,i] + cache[i,6:].tolist() + [amt])
    
    if i % 50000 == 0:
        print(count1,count2,count3,count4,count5,count6,count7,count8)
        print("*")
    
print(count1,count2,count3,count4,count5,count6,count7,count8)    
print("-")

cache = test_transaction[['uid','uid','ProductCD','TransactionAmt','addr1','TransactionDT'] + id_col + list(map(lambda x:x + "_day",id_col))].values
# ukey_dict = {}
# ukey2_dict = {}

for i in range(cache.shape[0]):
    uid = cache[i,0]
    isFraud = cache[i,1] 
    ProductCD = cache[i,2] 
    amt = (cache[i,3])
    addr1 = str(cache[i,4]) 
    
    t = cache[i,5] // 3600
    D1_day = float(cache[i,6]) 
    D1 = float(cache[i,7]) 
    match_id = ["0"] * len(id_col)     
    match_col = []
    
    ukey = uid + "_" + addr1
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[test_transaction.index[i]] = test_transaction.index[i]
    if len(ukey_dict[ukey]) >= 1:
        for j in range(min(15,len(ukey_dict[ukey]))):
            match_id = ["0"] * len(id_col)     
            match_col = []
            pos = -j - 1
            if abs(ukey_dict[ukey][pos][1] - t) > 3 * 24:
                continue
                
            isfit = False
            for j in range(len(id_col)):
                if cache[i,6+j+ len(id_col)] != '-1' and cache[i,6+j] >= 5 and abs(cache[i,6+j+ len(id_col)] - ukey_dict[ukey][pos][3+j + len(id_col)]) <= 1:
                    isfit = True
                    if True:
                        match_col.append((id_col[j],cache[i,6+j]))
                        match_id[j] = cache[i,6+j]
            if not isfit or len(list(filter(lambda x:x!="0",match_id)))<1:
                continue
                
            ukey2_dict[test_transaction.index[i]] = ukey2_dict[test_transaction.index[ukey_dict[ukey][pos][2]]]

                
            break
                    

    ukey_dict[ukey].append([isFraud,t,i] + cache[i,6:].tolist() + [amt])
    
    if i % 50000 == 0:
        print("*")

train_transaction['dkey'] = train_transaction.index.map(ukey2_dict)    
test_transaction['dkey'] = test_transaction.index.map(ukey2_dict)      


print(train_transaction.groupby('dkey')['TransactionAmt'].count().value_counts().iloc[:10])
print(test_transaction.groupby('dkey')['TransactionAmt'].count().value_counts().iloc[:10])
train_transaction[['dkey']].to_csv('../input/fg_train.csv',header = True)
test_transaction[['dkey']].to_csv('../input/fg_test.csv',header = True)