import os
import pandas as pd
import numpy as np
import random
from itertools import chain
import pickle

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath



class linearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

from trans import transform_w, tokenizer, get_trans, translate
weight_dir = os.path.join(os.getcwd(),'weights')
data_dir = os.path.join(os.getcwd(),'Data')
cn_embeddings = os.path.join(weight_dir, 'sgns.zhihu.bigram')
en_embeddings = os.path.join(weight_dir, 'word2vec-google-news-300.gz')
training_datapath = os.path.join(data_dir, 'transformation_training.csv')
pickle_path = os.path.join(data_dir, 'embedded_matrix_train.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Creating Training Data
"""
#
# cn_model = KeyedVectors.load_word2vec_format(datapath(cn_embeddings),
#                                              binary=False, unicode_errors="ignore")
# en_model = KeyedVectors.load_word2vec_format(datapath(en_embeddings), binary=True,
#                                              unicode_errors='ignore', limit=1000000)
# def get_embeddings(row):
#     en = row['en']
#     cn = row['cn']
#     if en in en_model.vocab:
#         row['en'] = en_model[en]
#     else:
#         print('en not in')
#         row['en'] = np.NaN
#     if cn in cn_model.vocab:
#         row['cn'] = cn_model[cn]
#     else:
#         print('cn not in')
#         row['cn'] = np.NaN
#     return row
#
# training_df = pd.read_csv(training_datapath, index_col=0)
# training_df.rename(columns={'en1': 'en', 'cn1': 'cn'}, inplace=True)
# print(training_df.shape)
#
# training_df = training_df.apply(get_embeddings, axis=1)
# print(training_df.shape)
# training_df.dropna(subset=['en','cn'], inplace=True)
# print(training_df.shape)
# train_data_cn = list(training_df['cn'])
# train_data_en = list(training_df['en'])

# with open(os.path.join(data_dir,'embedded_matrix_train_cn.pkl'), 'wb') as f:
#     pickle.dump(train_data_cn, f)
#
# with open(os.path.join(data_dir,'embedded_matrix_train_en.pkl'), 'wb') as f:
#     pickle.dump(train_data_en, f)

""""""""""""""""""""""""""
"""
Loading Training Data
"""

with open(os.path.join(data_dir, 'embedded_matrix_train_cn.pkl'), 'rb') as f:
    train_data_cn = pickle.load(f)
with open(os.path.join(data_dir, 'embedded_matrix_train_en.pkl'), 'rb') as f:
    train_data_en = pickle.load(f)
""""""""""""""""""""""""""

train_x, test_x, train_y, test_y = train_test_split(train_data_cn, train_data_en, test_size=0.3, random_state=17)


train_x_tensor = torch.tensor(train_x)
train_y_tensor = torch.tensor(train_y)
test_x_tensor = torch.tensor(test_x)
test_y_tensor = torch.tensor(test_y)

train_data = list(zip(train_x_tensor, train_y_tensor))
test_data = list(zip(test_x_tensor, test_y_tensor))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)

"""
Prepare Model
"""
model = linearRegression(300,300)
loss_fn = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.to(device)
model.train()
""""""""""""""""""""""""""

log_every_n = None
def train(train_loader, test_loader):
    def evaluate(loader):
        """
        Evaluate the model on a validation set
        """
        model.eval()
        batch_wise_true_labels = []
        batch_wise_predictions = []


        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader)):
                cn, en = batch
                logits = model(cn.to(device))

                predictions = logits

                batch_wise_true_labels.append(en)
                batch_wise_predictions.append(predictions.to('cpu'))


    loss_history = []
    running_loss = 0.
    running_loss_history = []
    for _ in tqdm(range(10)):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            cn, en = batch
            logits = model(cn.to(device))

            loss = loss_fn(logits, cn.to(device))
            loss_history.append(loss.item())
            running_loss += (loss_history[-1] - running_loss) / (i + 1)

            loss.backward()
            if log_every_n and i % log_every_n == 0:
                print("Running loss: ", running_loss)

            running_loss_history.append(running_loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print("Epoch completed!")
        print("Epoch Loss: ", running_loss)
        evaluate(test_loader)


train(train_loader, test_loader)
torch.save(model, os.path.join(data_dir,'transition_test.model'))