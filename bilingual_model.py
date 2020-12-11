import torch
from torch import nn
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import classification_report, f1_score
import math


class BiLSTMClassifier(nn.Module):
    def __init__(self, output_size,rnn_hidden_size=100, dropout_p=0.2, w2v_weights=None):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(w2v_weights, freeze=True)
        embed_dim = 300
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=rnn_hidden_size,
            bias=True,
            bidirectional=True,
            num_layers=1
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(rnn_hidden_size * 2, output_size)

    def forward(self, x, lengths):
        # lengths = torch.tensor(lengths).cpu()
        lengths = lengths.clone().detach()
        embed = x
        packed_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.rnn(packed_input)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        logits = self.fc(hidden)
        return logits

def evaluate(model, optimizer, loss_function, loader, device, labels, log_every_n=10):
    """
    Evaluate the model on a validation set
    """

    model.eval()

    batch_wise_true_labels = []
    batch_wise_predictions = []

    loss_history = []
    running_loss = 0.
    running_loss_history = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            input, labels, lengths = batch
            logits = model(input.to(device), lengths).squeeze()
            predictions = torch.sigmoid(logits)

            batch_wise_true_labels.append(labels.view(-1).tolist())
            batch_wise_predictions.append(predictions.view(-1).tolist())

    all_true_labels = list(chain.from_iterable(batch_wise_true_labels))
    all_predictions = list(chain.from_iterable(batch_wise_predictions))
    all_predictions = [1 if p > 0.5 else 0 for p in all_predictions]


    print("Evaluation Loss: ", running_loss)
    print("Classification report after epoch:")
    print(f1_score(all_true_labels, all_predictions, average='micro'))
    print(classification_report(all_true_labels, all_predictions, labels=labels))

    return loss_history, running_loss_history


def create_loader(inputs, transition_model, cn_model, en_model, device):
    '''Uses the tokenizer to tokenize the inputs and create a DataLoader'''
    lengths = [len(sequence[0]) for sequence in inputs]
    max_seq_len = max(lengths)

    data = []
    labels= []
    for i,seq in enumerate(inputs):
        sent, label = seq
        labels.append(label)
        vector = [[0 for _ in range(300)] for _ in range(max_seq_len)]
        for j,s in enumerate(sent):
            if s in cn_model.vocab:
                ce = cn_model[s]
                ce = torch.tensor(ce)
                with torch.no_grad():  # Disable gradient computation - required only during training
                    transform_embedding = transition_model(ce.to(device))
                    transform_embedding = transform_embedding.to('cpu')
                    transform_embedding = transform_embedding.numpy()
                    sim= en_model.similar_by_vector(transform_embedding)[0][0] # list of (word, similarity score)
                    vector[j] = en_model[sim] #word2vec returns a numpy array

        data.append(vector)
    inputs = torch.tensor(data)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return inputs, labels, lengths
