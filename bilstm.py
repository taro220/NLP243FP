import os

import torch
from torch import nn, optim

from sklearn.model_selection import train_test_split

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath

import pandas as pd

from preprocessing import W2VSequencer, tokzenize, TaggerDataset, prepare_batch
from bilstm_model import BiLSTMClassifier, run_training

"""

"""
BATCH_SIZE = 64
RANDOM_STATE = 17
hidden_size = 100
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Datasets
"""
# DATASET = 'datasetSentences.txt'
#
# DATASET = 'Office_Products_5.json.gz'
# DATASET = 'Movies_and_TV_5.json.gz'
#
DATASET = 'movies_even_split.csv'
# DATASET = 'Movies_and_TV_small.csv'
# DATASET = 'Movies_and_TV_small_mapped.csv'


dir_path = os.path.join(os.getcwd(), 'Data')
language_path = os.path.join(dir_path, 'English_Treebank')
dataset_path = os.path.join(language_path, DATASET)
weights_path = os.path.join(dir_path, 'word2vec-google-news-300.gz')

dataset_df = pd.read_csv(dataset_path)
# dataset_df = pd.read_json(dataset_path, lines=True)

# dataset_df['overall'] = dataset_df['overall'].apply(map_sentiment)
# sample_group = dataset_df.groupby('overall').sample(n=25000, random_state=1)
# sample_group[['overall', 'reviewText']].to_csv(os.path.join(language_path, 'movies_even_split.csv'))


# model = Word2Vec(sentences=dataset_df['reviewText'], window=5, min_count=1, workers=4)
# model.save('gensim.model')

word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format(datapath(weights_path), binary=True,
                                                                   unicode_errors='ignore', limit=1000000)
w2v_sequencer = W2VSequencer(word2vec_weights)

"""
Prepare Dataset
"""
dataset_df.dropna(inplace=True)
dataset_df = tokzenize(dataset_df)

train_X, val_X, train_y, val_y = train_test_split(dataset_df['tokenized'], dataset_df['overall'], random_state=RANDOM_STATE,
                                                  stratify=dataset_df['overall'], test_size=0.3)

train_data = list(zip(train_X, train_y))
train_dataset = TaggerDataset(train_data, w2v_sequencer)
val_data = list(zip(val_X, val_y))
val_dataset = TaggerDataset(val_data, w2v_sequencer)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           collate_fn=lambda batch: prepare_batch(batch, w2v_sequencer))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                         collate_fn=lambda batch: prepare_batch(batch, w2v_sequencer))

"""
Create Model
"""
lstm_clf = BiLSTMClassifier(output_size, rnn_hidden_size=hidden_size,
                            w2v_weights=torch.FloatTensor(word2vec_weights.vectors))

lstm_clf.to(device)

learning_rate = 1e-2
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_clf.parameters(), lr=learning_rate)


"""
Training
"""
run_training(lstm_clf, optimizer, loss_function, train_loader, val_loader, device, [0, 1], n_epochs=10)
