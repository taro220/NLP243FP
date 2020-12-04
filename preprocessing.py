import os
import numpy as np
import torch
from torch.utils.data import Dataset
import spacy

class TaggerDataset(Dataset):
    def __init__(self, data, input_sequencer):
        self.data = data
        self.input_sequencer = input_sequencer # Convert word tokens to list of integers

    def __getitem__(self, index):
        tokens, tags = self.data[index]

        x = self.input_sequencer.encode(tokens) # Input: [string], Output: [ints]
        y = tags # Input: [string], Output: [ints]

        return x, y

    def __len__(self):
        return len(self.data)

class W2VSequencer(object):
    """
    Class taken from Rishi's notebook
    """
    def __init__(self, gensim_w2v):

        self.w2v = gensim_w2v
        self.w2v.add(['<s>'], [np.zeros((300,))])
        self.w2v.add(['</s>'], [np.zeros((300,))])
        self.w2v.add(['<pad>'], [np.zeros((300,))])
        self.w2v.add(['<unk>'], [np.zeros((300,))])

        self.bos_index = self.w2v.vocab.get('<s>')
        self.eos_index = self.w2v.vocab.get('</s>')
        self.unk_index = self.w2v.vocab.get('<unk>')
        self.pad_index = self.w2v.vocab.get('<pad>')

    def encode(self, tokens):
        sequence = [self.bos_index.index]
        for token in tokens:
            index = self.w2v.vocab.get(token, self.unk_index).index
            sequence.append(index)
        sequence.append(self.eos_index.index)

        return sequence

    def create_padded_tensor_with_lengths(self, sequences):
        lengths = [len(sequence) for sequence in sequences]
        max_seq_len = max(lengths)
        tensor = torch.full((len(sequences), max_seq_len), self.pad_index.index, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor, lengths


def map_sentiment(val):
    if val > 3:
        return 1
    if val < 3:
        return 0
    else:
        return np.NaN

def prepare_dataset(df, language_path):
    df['overall'] = df['overall'].apply(map_sentiment)
    df.dropna(inplace=True)
    df[['reviewText', 'overall']].to_csv(os.path.join(language_path, 'Movies_and_TV_small_mapped.csv'), index=False)
    return df

def tokzenize(df):
    def helper(sent):
        tokens = [x.text for x in nlp.tokenizer(sent)]
        if len(tokens) > 300:
            return tokens[:300]
        return tokens
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    df['tokenized'] = df['reviewText'].apply(helper)
    return df


def prepare_batch(batch, in_sequencer):
    texts, labels = zip(*batch)
    text_tensor, lengths = in_sequencer.create_padded_tensor_with_lengths(texts)
    label_tensor = torch.tensor(labels)
    return (text_tensor, lengths, label_tensor)


# class Sequencer(object):
#     """
#     Class taken from Rishi's notebook
#     """
#     def __init__(self, tokens, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
#         self.word2idx = {}
#         self.idx2word = {}
#
#         self.pad_index = self.add_token(pad_token)
#         self.unk_index = self.add_token(unk_token)
#         self.bos_index = self.add_token(bos_token)
#         self.eos_index = self.add_token(eos_token)
#
#         for token in tokens:
#             self.add_token(token)
#
#     def add_token(self, token):
#
#         self.word2idx[token] = new_index = len(self.word2idx)
#         self.idx2word[new_index] = token
#
#         return new_index
#
#     def encode(self, tokens):
#         sequence = [self.bos_index]
#         for token in tokens:
#             index = self.word2idx.get(token, self.unk_index)
#             sequence.append(index)
#         sequence.append(self.eos_index)
#
#         return sequence
#
#     def create_padded_tensor_with_lengths(self, sequences):
#         lengths = [len(sequence) for sequence in sequences]
#         max_seq_len = max(lengths)
#         tensor = torch.full((len(sequences), max_seq_len), self.pad_index, dtype=torch.long)
#
#         for i, sequence in enumerate(sequences):
#             for j, token in enumerate(sequence):
#                 tensor[i][j] = token
#
#         return tensor, lengths