import torch
from torch import nn
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import classification_report, f1_score
import math


class BiLSTMClassifier(nn.Module):
    def __init__(self, output_size,rnn_hidden_size=100, dropout_p=0.5, w2v_weights=None):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(w2v_weights, freeze=False)
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
        lengths = torch.tensor(lengths).cpu()

        embed = self.dropout(self.embedding(x))

        packed_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.rnn(packed_input)

        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        logits = self.fc(hidden)

        return logits

def train(model, optimizer, loss_function, loader, device, log_every_n=10):
    """
    Run a single epoch of training
    """

    model.train()  # Run model in training mode

    loss_history = []
    running_loss = 0.
    running_loss_history = []

    for i, batch in tqdm(enumerate(loader)):

        optimizer.zero_grad()  # Always set gradient to 0 before computing it

        logits = model(batch[0].to(device), batch[1]).squeeze()

        loss = loss_function(logits, batch[2].to(device))

        loss_history.append(loss.item())
        running_loss += (loss_history[-1] - running_loss) / (i + 1)  # Compute rolling average

        loss.backward()  # Perform backprop, which will compute dL/dw

        if log_every_n and i % log_every_n == 0:
            print("Running loss: ", running_loss)

        running_loss_history.append(running_loss)

        nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # We clip gradient's norm to 3

        optimizer.step()  # Update step: w = w - eta * dL / dW : eta = 1e-2 (0.01), gradient = 5e30; update value of 5e28

    print("Epoch completed!")
    print("Epoch Loss: ", running_loss)
    print("Epoch Perplexity: ", math.exp(running_loss))

    # The history information can allow us to draw a loss plot
    return loss_history, running_loss_history


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

    with torch.no_grad():  # Disable gradient computation - required only during training
        for i, batch in tqdm(enumerate(loader)):

            logits = model(batch[0].to(device), batch[1]).squeeze()
            loss = loss_function(logits, batch[2].to(device))
            loss_history.append(loss.item())

            running_loss += (loss_history[-1] - running_loss) / (i + 1)  # Compute rolling average

            running_loss_history.append(running_loss)

            predictions = torch.sigmoid(logits)

            batch_wise_true_labels.append(batch[2].view(-1).tolist())
            batch_wise_predictions.append(predictions.view(-1).tolist())

    # flatten the list of predictions using itertools
    all_true_labels = list(chain.from_iterable(batch_wise_true_labels))
    all_predictions = list(chain.from_iterable(batch_wise_predictions))
    all_predictions = [1 if p > 0.5 else 0 for p in all_predictions]


    print("Evaluation Loss: ", running_loss)
    # Now we can generate a classification report
    print("Classification report after epoch:")
    print(f1_score(all_true_labels, all_predictions, average='micro'))
    print(classification_report(all_true_labels, all_predictions, labels=labels))

    return loss_history, running_loss_history

def run_training(model, optimizer, loss_function, train_loader, valid_loader, device, labels, n_epochs=10):
    for i in range(n_epochs):
        train(model, optimizer, loss_function, train_loader, device, log_every_n=10)
        evaluate(model, optimizer, loss_function, valid_loader, device, labels, log_every_n=10)

    torch.save(model,'en_model.pkl')
    torch.save(model.state_dict(), 'en_model_state.pkl')
