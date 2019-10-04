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

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card4'].astype(str)
test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card4'].astype(str)


train_test = train_transaction.append(test_transaction)
train_test['dist1'] = train_test['dist1'].fillna(-1)
train_test['addr1'] = train_test['addr1'].fillna(-1)



cache = train_test[['TransactionDT','TransactionAmt','uid','addr1','dist1']].values
isFraud = train_test[['isFraud']].values




card1_addr1_dist = {}
f1 = open('../input/fa_deepwalk.csv','w')
f = f1
index_feature = {}
index = 0
import pickle
# maxnums = 200000
# maxlength = 128
for i in range(cache.shape[0]):
    time = "TransactionDT_" + str(int(cache[i,0]//(3600*24*7)))
    addr1_dist = "addr1_dist1_" + str((cache[i,3])) + "_" + str((cache[i,4]))
    card1 = "card1_" + str((cache[i,2]))
    if "-1" in addr1_dist:
        continue
    
    

        
    if card1_addr1_dist.get(card1,-1) != -1:
        if cache[i,0] - card1_addr1_dist[card1][1] < 3600 * 24:
            if index_feature.get(addr1_dist,-1) == -1:
                index_feature[addr1_dist] = index
                index += 1 
            if index_feature.get(card1_addr1_dist[card1][0],-1) == -1:
                index_feature[card1_addr1_dist[card1][0]] = index
                index += 1
            print("\t".join([str(index_feature[addr1_dist]),str(index_feature[card1_addr1_dist[card1][0]])]),file=f)
    
    card1_addr1_dist[card1] = (addr1_dist,cache[i,0])

f1.close()
print(len(index_feature))
with open('emb3/index_feature.pkl', 'wb') as f:
    pickle.dump(index_feature, f)   


import os
os.system("python proNE.py -graph ../input/fa_deepwalk.csv -emb1 emb3/sparse.emb -emb2 emb3/spectral.emb -dimension 16 -step 5 -theta 0.5 -mu 0.2")
    
# import logging
# import gensim
# from gensim.models import Word2Vec


# dim = 8
# min_count = 2

# dump_name = './f3_w2v.{}.{}.gensim.txt'.format( dim, min_count)
# seg_corpus = []

# for line in open("../input/f3_deepwalk.csv"):
#     if not line:
#         continue
#     seg_corpus.append(line.strip().split())

# print(len(seg_corpus))
# print(" ".join(seg_corpus[0]))
# word2vec = Word2Vec(seg_corpus, size=dim, min_count=min_count, sg=1, hs=0, negative=10, iter=5, workers=6, window=5)
# word2vec.wv.save_word2vec_format(dump_name)