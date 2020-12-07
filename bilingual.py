import os
from itertools import chain
# import translators as ts
from translate import Translator
import numpy as np
import random
import pickle
import torch
from torch import nn, optim
from linear_regression_model import linearRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
import gensim.downloader as api
from bilingual_model import evaluate, BiLSTMClassifier, create_loader

import pandas as pd

from preprocessing import W2VSequencer, TaggerDataset, prepare_batch
from bilstm_model import run_training

from trans import transform_w, tokenizer, get_trans, translate


data_dir = os.path.join(os.getcwd(),'Data')
cn_embeddings = os.path.join(data_dir, 'sgns.zhihu.bigram')
en_embeddings = os.path.join(data_dir, 'word2vec-google-news-300.gz')
cn_test1 = os.path.join(data_dir, 'jd_xiaomi9_neg.csv')
cn_test2 = os.path.join(data_dir, 'jd_xiaomi9_pos.csv')

transition_model_path = os.path.join(data_dir, 'transition2.model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
cn_model = KeyedVectors.load_word2vec_format(datapath(cn_embeddings),
                                             binary=False, unicode_errors="ignore")
en_model = KeyedVectors.load_word2vec_format(datapath(en_embeddings), binary=True,
                                             unicode_errors='ignore', limit=1000000)


test_pos_df = pd.read_csv(cn_test1, encoding='GBK')
test_neg_df = pd.read_csv(cn_test2, encoding='GBK')
transition_model = torch.load(transition_model_path)
transition_model.to(device)
transition_model.eval()

# def get_embeddings(sent):
#     vector = []
#     for s in sent:
#         if s in cn_model.vocab:
#             ce = cn_model[s]
#             ce = torch.tensor(ce)
#             with torch.no_grad():  # Disable gradient computation - required only during training
#                 transform_embedding = transition_model(ce.to(device))
#             transform_embedding = transform_embedding.to('cpu')
#             transform_embedding = transform_embedding.numpy()
#             sim= en_model.similar_by_vector(transform_embedding)[0][0] # list of (word, similarity score)
#             vector.append(torch.from_numpy(en_model[sim])) #word2vec returns a numpy array
#         else:
#             print('cn not in')
#             sent = np.NaN
#         return vector


test_df = test_pos_df.append(test_neg_df)
# test_set = list(test_df['content'].apply(tokenizer))
# small_test = test_set[:50]
# test_labels = list(small_test['score'])
# small_test = list(small_test['content'])
# small_test = small_test['content'].apply(get_embeddings)
# print(small_test)
w2v_sequencer = W2VSequencer(en_model)
# #
#
test_df = test_df.sample(n=50, random_state=17)
test_labels = test_df['score'].apply(lambda x: 0 if x < 3 else 1)
test_labels = list(test_labels)
test_sent = test_df['content'].apply(tokenizer)
test_sent = list(test_sent)

testing_dataset = list(zip(test_sent, test_labels))

# train_dataset = TaggerDataset(testing_dataset, w2v_sequencer)

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=4,
                                           collate_fn=lambda batch: create_loader(batch, transition_model, cn_model, en_model, device))

# (inputs, labels, transition_model, cn_model, en_model, device):

# model = torch.load('en_model.pkl')
# model.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cn_model.vectors), freeze=False)
model = BiLSTMClassifier(1, rnn_hidden_size=100,
                            w2v_weights=torch.FloatTensor(en_model.vectors))
model.load_state_dict(torch.load('en_model_state.pkl'))

model.eval()
model.to(device)

# def evaluate(model, optimizer, loss_function, loader, device, labels, log_every_n=10):
evaluate(model, None, transition_model, testing_loader, device, None, None)
#
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
#                                          collate_fn=lambda batch: prepare_batch(batch, w2v_sequencer))
#
# w2v_sequencer = W2VSequencer(en_model)
# train_data = list(zip(train_X, train_y))
# train_dataset = TaggerDataset(train_data, w2v_sequencer)
#
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
#                                            collate_fn=lambda batch: prepare_batch(batch, w2v_sequencer))
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
#                                          collate_fn=lambda batch: prepare_batch(batch, w2v_sequencer))
#
#
#
#



#
#

# cn = cn_model['汽车']
# cn = torch.tensor(cn)
# with torch.no_grad():  # Disable gradient computation - required only during training
#     transform_embedding = transition_model(cn.to(device))
# transform_embedding= transform_embedding.to('cpu')
# transform_embedding = transform_embedding.numpy()
# # print(en_model.similar_by_vector(transform_embedding))
# print(en_model.similar_by_vector(transform_embedding))