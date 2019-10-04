import os

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')
train_pred['TransactionDT']=train_transaction['TransactionDT']
test_pred['TransactionDT']=test_transaction['TransactionDT']
test_pred['isFraud'] = -1


# train_test['card1'] = train_test['card1'].fillna(-1)
# train_test['card2'] = train_test['card2'].fillna(-1)



def encode_vt(train, vali, id_cate, target, use_bayes_smooth):
    col_name = '_'.join(id_cate) + '_cvr'
    col_name2 = '_'.join(id_cate) + '_sum'
    col_name3 = '_'.join(id_cate) + '_size'
    grouped = train.groupby(id_cate, as_index=False)[target].agg({"C": "size", "V": "sum"})
    C = grouped['C']
    V = grouped['V']
    if (use_bayes_smooth):
        print('start smooth')
        hyper = HyperParam(1, 1)
        hyper.update_from_data(C, V)
        print('end smooth')
        grouped[col_name] = (hyper.alpha + V) / (hyper.alpha + hyper.beta + C)
    else:
#         grouped[col_name] = ((V / C) + 0.375 * (2.0 - C.map(lambda x:min(x,10.0)) * 0.1))/(1 + (2.0 - C.map(lambda x:min(x,10.0)) * 0.1))
        grouped[col_name] = (V / C)
        grouped[col_name2] = V
        grouped[col_name3] = C
    grouped[col_name] = grouped[col_name].astype('float32')
    grouped[col_name2] = grouped[col_name2].astype('float32')
    grouped[col_name3] = grouped[col_name3].astype('float32')
    df = vali[id_cate].merge(grouped, 'left', id_cate)[[col_name,col_name2,col_name3]]
    df = np.asarray(df, dtype=np.float32)
    del grouped
    return df

col_14 = [['card1'], ['card2'], ['TransactionAmt'], ['addr1']]
target = 'isFraud'
use_bayes_smooth = 0
col_var = []
vali = test_transaction
train = train_transaction
for id_cate in col_14:
    print(id_cate)
    col_name = '_'.join(id_cate) + '_cvr'
    col_name2 = '_'.join(id_cate) + '_sum'
    col_name3 = '_'.join(id_cate) + '_size'
    col_var.append(col_name)
    col_var.append(col_name2)
    col_var.append(col_name3)
    bayes_feature = encode_vt(train, vali, id_cate, target, use_bayes_smooth)
    vali[col_name] = 0
    vali[col_name2] = 0
    vali[col_name3] = 0
    vali[[col_name,col_name2,col_name3]] = bayes_feature
    skf = KFold(n_splits=5, shuffle=False, random_state=10).split(train[target].values)
    train[col_name] = 0
    train[col_name2] = 0
    train[col_name3] = 0

    for train_idx, test_idx in skf:
        # print(id_cate, target)
        X_train = train.iloc[train_idx]
        X_test = train.iloc[test_idx]
        bayes_feature = encode_vt(X_train, X_test, id_cate, target, use_bayes_smooth)
        train.ix[train.iloc[test_idx].index, [col_name,col_name2,col_name3]] = bayes_feature
        del bayes_feature


