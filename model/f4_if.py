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
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

# train_transaction['hour'] = train_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
# test_transaction['hour'] = test_transaction['TransactionDT'].map(lambda x:(x//3600)%24)
# train_transaction['weekday'] = train_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
# test_transaction['weekday'] = test_transaction['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
# train_transaction['TransactionAmt_decimal'] = ((train_transaction['TransactionAmt'] - train_transaction['TransactionAmt'].astype(int)) * 1000).astype(int)
# test_transaction['TransactionAmt_decimal'] = ((test_transaction['TransactionAmt'] - test_transaction['TransactionAmt'].astype(int)) * 1000).astype(int)

from sklearn.linear_model import LinearRegression 
train_test = train_transaction.append(test_transaction)
print(train_test.columns)
# for col in "D1,D2,D3,D4,D5,D6,D7,D8,D10,D11,D12,D13,D14,D15".split(","):
#     df = train_test[['TransactionDT',col]]
#     df = df[~df[col].isna()][df[col]>50.0]
#     x = np.asarray(df[['TransactionDT']])*0.0001
#     y = np.asarray(df[[col]])
#     reg = LinearRegression().fit(x, y)
#     print(train_test[col].max(),train_test[col].min())
#     print(col," Y = %.5fX + (%.5f)" % (reg.coef_[0][0], reg.intercept_[0]),x[0]*reg.coef_[0][0] + reg.intercept_[0],x[0],y[0])
#     train_transaction[col+'_fix'] = train_transaction[col].fillna(-1)*reg.intercept_[0]/(train_transaction['TransactionDT'].map(lambda x:x *0.0001 * reg.coef_[0][0]) + reg.intercept_[0])
#     train_transaction[col+'_fix'] = train_transaction[col+'_fix'].map(lambda x:-0.1 if x<0 else x)
#     test_transaction[col+'_fix'] = test_transaction[col].fillna(-1)*reg.intercept_[0]/(test_transaction['TransactionDT'].map(lambda x:x *0.0001 * reg.coef_[0][0]) + reg.intercept_[0])
#     test_transaction[col+'_fix'] = test_transaction[col+'_fix'].map(lambda x:-0.1 if x<0 else x)

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)


test_transaction['isFraud'] = 0
y_train = train_transaction['isFraud'].copy()
y_test = test_transaction['isFraud'].copy()
X_train = train_transaction.drop('isFraud', axis=1).fillna(-999)
X_test = test_transaction.drop('isFraud', axis=1).fillna(-999)


#         del train_transaction[col],test_transaction[col]
# X_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True).fillna(-999)
# X_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True).fillna(-999)



for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))
        
clf = IsolationForest(n_estimators=100, max_samples='auto', random_state=10, n_jobs=6)
clf.fit(X_train.append(X_test))

debug = False
if debug:
    split_pos = X_train.shape[0]*4//5
    y_test = y_train.iloc[split_pos:]
    y_train = y_train.iloc[:split_pos]
    X_test = X_train.iloc[split_pos:,:]
    X_train = X_train.iloc[:split_pos,:]

   
    
# fit the model

y_pred_train = clf.decision_function(X_train)
y_pred_test = clf.decision_function(X_test)

# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)

print(y_pred_train.shape,y_pred_train.sum())
print(y_pred_test.shape,y_pred_test.sum())
print(roc_auc_score(y_train,1-y_pred_train))
if debug:
    print(roc_auc_score(y_test,1-y_pred_test))
# y_pred_train = clf.score_samples(X_train)
# y_pred_test = clf.score_samples(X_test)

# # y_pred_train = clf.predict(X_train)
# # y_pred_test = clf.predict(X_test)

# print(y_pred_train.shape,y_pred_train.sum())
# print(y_pred_test.shape,y_pred_test.sum())
# print(roc_auc_score(y_train,1-y_pred_train))
# print(roc_auc_score(y_test,1-y_pred_test))
if True:
    X_train_pred = pd.DataFrame(index = X_train.index)
    X_train_pred['iso'] = y_pred_train
    X_train_pred.to_csv('../input/f4_train.csv')
    X_test_pred = pd.DataFrame(index = X_test.index)
    X_test_pred['iso'] = y_pred_test
    X_test_pred.to_csv('../input/f4_test.csv')

for i in range(339):
    col = "V" + str(i+1)
    s = train_transaction[col].fillna(0).map(lambda x:0 if x%1 == 0 else 1).sum()
    if s > 100:
        if (i + 1<263 and i + 1 >= 215) or i + 1 < 159 or i + 1 >= 276:
            train_transaction[col] = train_transaction[col]/train_transaction['TransactionAmt']
            test_transaction[col] = test_transaction[col]/test_transaction['TransactionAmt']
        del train_transaction[col],test_transaction[col]
    
# X_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True).fillna(-999)
# X_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True).fillna(-999)
X_train = train_transaction.drop('isFraud', axis=1).fillna(-999)
X_test = test_transaction.drop('isFraud', axis=1).fillna(-999)


for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))
        
clf = IsolationForest(n_estimators=100, max_samples='auto', random_state=10, n_jobs=6)
clf.fit(X_train.append(X_test))


   
    

y_pred_train = clf.decision_function(X_train)
y_pred_test = clf.decision_function(X_test)
print(roc_auc_score(y_train,1-y_pred_train))
if debug:
    print(roc_auc_score(y_test,1-y_pred_test))
X_train_pred['iso2'] = y_pred_train
X_test_pred['iso2'] = y_pred_test
X_train_pred.to_csv('../input/f4_train.csv')
X_test_pred.to_csv('../input/f4_test.csv')