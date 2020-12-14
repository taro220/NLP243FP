import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from bilingual_model import evaluate, BiLSTMClassifier, create_loader
from preprocessing import W2VSequencer
from trans import tokenizer
from linear_regression_model import linearRegression #Dont remove this line

data_dir = os.path.join(os.getcwd(),'Data')
weight_dir = os.path.join(os.getcwd(),'weights')
cn_embeddings = os.path.join(weight_dir, 'sgns.zhihu.bigram')
en_embeddings = os.path.join(weight_dir, 'word2vec-google-news-300.gz')
cn_test1 = os.path.join(data_dir, 'jd_xiaomi9_neg.csv')
cn_test2 = os.path.join(data_dir, 'jd_xiaomi9_pos.csv')
en_model_weights = os.path.join(weight_dir, 'en_model_state.pkl')

transition_model_path = os.path.join(weight_dir, 'transition.model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Word Embeddings
"""
cn_model = KeyedVectors.load_word2vec_format(datapath(cn_embeddings),
                                             binary=False, unicode_errors="ignore")
en_model = KeyedVectors.load_word2vec_format(datapath(en_embeddings), binary=True,
                                             unicode_errors='ignore', limit=1000000)
""""""""""""""""""""""""""
"""
Test Data
"""
test_pos_df = pd.read_csv(cn_test1, encoding='GBK')
test_neg_df = pd.read_csv(cn_test2, encoding='GBK')

test_df = test_pos_df.append(test_neg_df)
w2v_sequencer = W2VSequencer(en_model)

test_df = test_df.sample(n=100, random_state=17)
test_labels = test_df['score'].apply(lambda x: 0 if x < 3 else 1)
test_labels = list(test_labels)
test_sent = test_df['content'].apply(tokenizer)
test_sent = list(test_sent)

testing_dataset = list(zip(test_sent, test_labels))

testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=4,
                                           collate_fn=lambda batch: create_loader(batch, transition_model, cn_model, en_model, device))

""""""""""""""""""""""""""
"""
Model Prep
"""
transition_model = torch.load(transition_model_path)
transition_model.to(device)
transition_model.eval()

model = BiLSTMClassifier(1, rnn_hidden_size=100,
                            w2v_weights=torch.FloatTensor(en_model.vectors))

model.load_state_dict(torch.load(en_model_weights))

model.eval()
model.to(device)
""""""""""""""""""""""""""

evaluate(model, None, transition_model, testing_loader, device, None, None)
