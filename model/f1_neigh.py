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
# train_pred = pd.read_csv('./simple_xgboost_cv.csv')
# test_pred = pd.read_csv('./simple_xgboost_pred.csv')
# train_pred['TransactionDT']=train_transaction['TransactionDT']
# test_pred['TransactionDT']=test_transaction['TransactionDT']
# test_pred['pred'] = (test_pred['pred1'] + test_pred['pred2'] + test_pred['pred3'] )/3
# del test_pred['pred1'],test_pred['pred2'],test_pred['pred3']
# test_pred['isFraud'] = -1


train_test = train_transaction.append(test_transaction)
train_test['card1'] = train_test['card1'].fillna(-1)
train_test['card2'] = train_test['card2'].fillna(-1)

train_transaction.drop([x for x in train_transaction.columns if x not in  ['TransactionID']],axis = 1,inplace = True)
test_transaction.drop([x for x in test_transaction.columns if x not in  ['TransactionID']],axis = 1,inplace = True)


# TransactionID_pred = {}
# cache = train_test[['TransactionID','pred']].values
# for i in range(cache.shape[0]):
#     id = int(cache[i,0])
#     pred = cache[i,1]
#     TransactionID_pred[id] = pred


cache = train_test[['TransactionDT','TransactionAmt','card1','card2','TransactionID']].values
isFraud = train_test[['isFraud']].values



time_amt = {}  # one index
time_card1 = {}
time_card2 = {}
time_amt_card1 = {} # two index
time_card1_amt = {}
card1_time = {}
card2_time = {}
amt_time = {}
for i in range(cache.shape[0]):
    time = int(cache[i,0]//3600)
    amt = cache[i,1]
    card1 = cache[i,2]
    card2 = cache[i,3]
    
    time_amt[time] = time_amt.get(time,{})
    time_card1[time] = time_card1.get(time,{})
    time_card2[time] = time_card2.get(time,{})
    time_amt_card1[time] = time_amt_card1.get(time,{})
    time_card1_amt[time] = time_card1_amt.get(time,{})
    
    time_amt[time][amt] = time_amt[time].get(amt,0) + 1
    time_card1[time][card1] = time_card1[time].get(card1,0) + 1
    time_card2[time][card2] = time_card2[time].get(card2,0) + 1
    
    time_amt_card1[time][amt] = time_amt_card1[time].get(amt,{})
    time_amt_card1[time][amt][card1] = time_amt_card1[time][amt].get(card1,0) + 1
    time_card1_amt[time][card1] = time_card1_amt[time].get(card1,{})
    time_card1_amt[time][card1][amt] = time_card1_amt[time][card1].get(amt,0) + 1
    
    id = int(cache[i,4])
#     card1_time[card1] = card1_time.get(card1,[])
#     card2_time[card2] = card2_time.get(card2,[])
#     amt_time[amt] = amt_time.get(amt,[])
    card1_time[card1] = card1_time.get(card1,{})
    card1_time[card1][time] = card1_time[card1].get(time,[])
    card1_time[card1][time].append(id)
    card2_time[card2] = card2_time.get(card2,{})
    card2_time[card2][time] = card2_time[card2].get(time,[])
    card2_time[card2][time].append(id)
    amt_time[amt] = amt_time.get(amt,{})
    amt_time[amt][time] = amt_time[amt].get(time,[])
    amt_time[amt][time].append(id)
    
    if i % 100000 == 0:
        print("*")
    
print("-")

amt_counts = []
amt_card1_counts = []
card1_counts = []
card1_amt_counts = []
card2_counts = []

amt_means = []
amt_maxs = []
amt_10_counts = []
card1_means = []
card1_maxs = []
card1_10_counts = []
card2_means = []
card2_maxs = []
card2_10_counts = []


f1 = open('../input/graph_train.csv','w')
f2 = open('../input/graph_test.csv','w')
for i in range(cache.shape[0]):
    time = int(cache[i,0]//3600)
    amt = cache[i,1]
    card1 = cache[i,2]
    card2 = cache[i,3]
    id = int(cache[i,4])
    
    amt_count = 0
    card1_count = 0
    card2_count = 0
    amt_card1_distinct_count = 0
    card1_amt_distinct_count = 0
    for t in range(time - 24,time + 24):
        amt_count += time_amt.get(t,{}).get(amt,0)
        card1_count += time_card1.get(t,{}).get(card1,0)
        card2_count += time_card2.get(t,{}).get(card2,0)
        
        amt_card1_distinct_count += len(time_amt_card1.get(t,{}).get(amt,{}))
        card1_amt_distinct_count += len(time_card1_amt.get(t,{}).get(card1,{}))
            
            
    amt_counts.append(amt_count - 1)
    card1_counts.append(card1_count - 1)
    card2_counts.append(card2_count - 1)
    amt_card1_counts.append(amt_card1_distinct_count - 1)
    card1_amt_counts.append(card1_amt_distinct_count - 1)
    
    
    # graph
    
    # debug
    if i >= train_transaction.shape[0]:
        f = f2
    else:
        f = f1
    

    
    amt_list = []
    card1_list = []
    card2_list = []
#     for t in range(time - 24,time + 24):
#         for dt in card1_time.get(card1,{}).get(t,[]):
#             if dt != id:
#                 card1_list.append(dt)
#         for dt in card2_time.get(card2,{}).get(t,[]):
#             if dt != id:
#                 card2_list.append(dt)
#         for dt in amt_time.get(amt,{}).get(t,[]):
#             if dt != id:
#                 amt_list.append(dt)
    
#     amt_means.append(0 if (len(amt_list) == 0) else np.mean([TransactionID_pred[x] for x in amt_list]))
#     amt_maxs.append(0 if (len(amt_list) == 0) else np.max([TransactionID_pred[x] for x in amt_list]))
#     amt_10_counts.append(len([x for x in amt_list if TransactionID_pred[x] > 0.1]))
#     card1_means.append(0 if (len(card1_list) == 0) else np.mean([TransactionID_pred[x] for x in card1_list]))
#     card1_maxs.append(0 if (len(card1_list) == 0) else np.max([TransactionID_pred[x] for x in card1_list]))
#     card1_10_counts.append(len([x for x in card1_list if TransactionID_pred[x] > 0.1]))
#     card2_means.append(0 if (len(card2_list) == 0) else np.mean([TransactionID_pred[x] for x in card2_list]))
#     card2_maxs.append(0 if (len(card2_list) == 0) else np.max([TransactionID_pred[x] for x in card2_list]))
#     card2_10_counts.append(len([x for x in card2_list if TransactionID_pred[x] > 0.1]))
    
    print(str(isFraud[i,0])+","+str(id)+","+" ".join([str(x) for x in amt_list]) + "," + " ".join([str(x) for x in card1_list]) + "," + " ".join([str(x) for x in card2_list]),file=f)
    if i % 100000 == 0:
        print("*")


    
        
s = train_transaction.shape[0]
train_transaction['f1_amt_counts'] = np.array(amt_counts[:train_transaction.shape[0]])
test_transaction['f1_amt_counts'] = np.array(amt_counts[train_transaction.shape[0]:])
train_transaction['f1_amt_card1_counts'] = np.array(amt_card1_counts[:train_transaction.shape[0]])
test_transaction['f1_amt_card1_counts'] = np.array(amt_card1_counts[train_transaction.shape[0]:])
train_transaction['f1_card1_counts'] = np.array(card1_counts[:train_transaction.shape[0]])
test_transaction['f1_card1_counts'] = np.array(card1_counts[train_transaction.shape[0]:])
train_transaction['f1_card1_amt_counts'] = np.array(card1_amt_counts[:train_transaction.shape[0]])
test_transaction['f1_card1_amt_counts'] = np.array(card1_amt_counts[train_transaction.shape[0]:])
train_transaction['f1_card2_counts'] = np.array(card2_counts[:train_transaction.shape[0]])
test_transaction['f1_card2_counts'] = np.array(card2_counts[train_transaction.shape[0]:])

amt_counts = []
amt_card1_counts = []
card1_counts = []
card1_amt_counts = []
card2_counts = []
dt = 48
for i in range(cache.shape[0]):
    time = int(cache[i,0]//3600)
    amt = cache[i,1]
    card1 = cache[i,2]
    card2 = cache[i,3]
    id = int(cache[i,4])
    
    amt_count = 0
    card1_count = 0
    card2_count = 0
    amt_card1_distinct_count = 0
    card1_amt_distinct_count = 0
    for t in range(time - dt,time + dt):
        amt_count += time_amt.get(t,{}).get(amt,0)
        card1_count += time_card1.get(t,{}).get(card1,0)
        card2_count += time_card2.get(t,{}).get(card2,0)
        
        amt_card1_distinct_count += len(time_amt_card1.get(t,{}).get(amt,{}))
        card1_amt_distinct_count += len(time_card1_amt.get(t,{}).get(card1,{}))
            
            
    amt_counts.append(amt_count - 1)
    card1_counts.append(card1_count - 1)
    card2_counts.append(card2_count - 1)
    amt_card1_counts.append(amt_card1_distinct_count - 1)
    card1_amt_counts.append(card1_amt_distinct_count - 1)

s = train_transaction.shape[0]
train_transaction['f1' + str(dt)+ '_amt_counts'] = np.array(amt_counts[:train_transaction.shape[0]])
test_transaction['f1'+str(dt)+'_amt_counts'] = np.array(amt_counts[train_transaction.shape[0]:])
train_transaction['f1'+str(dt)+'_amt_card1_counts'] = np.array(amt_card1_counts[:train_transaction.shape[0]])
test_transaction['f1'+str(dt)+'_amt_card1_counts'] = np.array(amt_card1_counts[train_transaction.shape[0]:])
train_transaction['f1'+str(dt)+'_card1_counts'] = np.array(card1_counts[:train_transaction.shape[0]])
test_transaction['f1'+str(dt)+'_card1_counts'] = np.array(card1_counts[train_transaction.shape[0]:])
train_transaction['f1'+str(dt)+'_card1_amt_counts'] = np.array(card1_amt_counts[:train_transaction.shape[0]])
test_transaction['f1'+str(dt)+'_card1_amt_counts'] = np.array(card1_amt_counts[train_transaction.shape[0]:])
train_transaction['f1'+str(dt)+'_card2_counts'] = np.array(card2_counts[:train_transaction.shape[0]])
test_transaction['f1'+str(dt)+'_card2_counts'] = np.array(card2_counts[train_transaction.shape[0]:])
train_transaction.to_csv('../input/f1_train.csv',index = None)
test_transaction.to_csv('../input/f1_test.csv',index = None)

# train_transaction['amt_means'] = np.array(amt_means[:s])
# test_transaction['amt_means'] = np.array(amt_means[s:])
# train_transaction['amt_maxs'] = np.array(amt_maxs[:s])
# test_transaction['amt_maxs'] = np.array(amt_maxs[s:])
# train_transaction['amt_10_counts'] = np.array(amt_10_counts[:s])
# test_transaction['amt_10_counts'] = np.array(amt_10_counts[s:])
# train_transaction['card1_means'] = np.array(card1_means[:s])
# test_transaction['card1_means'] = np.array(card1_means[s:])
# train_transaction['card1_maxs'] = np.array(card1_maxs[:s])
# test_transaction['card1_maxs'] = np.array(card1_maxs[s:])
# train_transaction['card1_10_counts'] = np.array(card1_10_counts[:s])
# test_transaction['card1_10_counts'] = np.array(card1_10_counts[s:])
# train_transaction['card2_means'] = np.array(card2_means[:s])
# test_transaction['card2_means'] = np.array(card2_means[s:])
# train_transaction['card2_maxs'] = np.array(card2_maxs[:s])
# test_transaction['card2_maxs'] = np.array(card2_maxs[s:])
# train_transaction['card2_10_counts'] = np.array(card2_10_counts[:s])
# test_transaction['card2_10_counts'] = np.array(card2_10_counts[s:])
# train_transaction[['TransactionID','isFraud','amt_means','amt_maxs','amt_10_counts','card1_means','card1_maxs','card1_10_counts','card2_means','card2_maxs','card2_10_counts']].to_csv('../input/f2_train.csv',index = None)
# test_transaction[['TransactionID','amt_means','amt_maxs','amt_10_counts','card1_means','card1_maxs','card1_10_counts','card2_means','card2_maxs','card2_10_counts']].to_csv('../input/f2_test.csv',index = None)


