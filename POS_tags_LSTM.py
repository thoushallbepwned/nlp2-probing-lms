from collections import defaultdict
from lstm.model import RNNModel
import torch
from typing import List
import torch
from conllu import parse_incr, TokenList
from torch import Tensor
import pickle
from tqdm import tqdm
from sklearn import preprocessing
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os
import random

model_location = 'state_dict.pt'  # <- point this to the location of the Gulordava .pt file
lstm = RNNModel('LSTM', 50001, 650, 650, 2)
lstm.load_state_dict(torch.load(model_location))

with open('lstm/vocab.txt', encoding="utf8") as f:
    w2i = {w.strip(): i for i, w in enumerate(f)}

vocab = defaultdict(lambda: w2i["<unk>"])
vocab.update(w2i)


# If stuff like `: str` and `-> ..` seems scary, fear not! 
# These are type hints that help you to understand what kind of argument and output is expected.
def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

    
def fetch_sen_reps_lstm(ud_parses: List[TokenList], model, tokenizer) -> Tensor:
    representation_size = 650
    out = []
    for sentence in ud_parses:
        sent = torch.zeros((len(sentence), 1))
        for i, token in enumerate(sentence):
            sent[i] = tokenizer[str(token)]
        sent = sent.long()
        hidden = model.init_hidden(len(sentence))
        with torch.no_grad():
            model.eval()
            out_sentence = model(sent, hidden)

        out += out_sentence.squeeze(1)
    out = torch.stack(out)
    return out    

def fetch_pos_tags(ud_parses: List[TokenList], pos_vocab=None) -> Tensor:
    pos_tags = list()
    for sentence in ud_parses:
        for token in sentence:

            pos_tags.append(token["upostag"])

    if pos_vocab:
        targets = pos_vocab.fit_transform(pos_tags)
        pos_tags = torch.as_tensor(targets)
    else:
        pos_vocab = preprocessing.LabelEncoder()
        targets = pos_vocab.fit_transform(pos_tags)
        pos_tags = torch.as_tensor(targets)
  
    return pos_tags, pos_vocab

def fetch_pos_tags_control(ud_parses: List[TokenList], pos_vocab=None) -> Tensor:
    pos_tags = list()
    if pos_vocab:
        pos_vocab = pos_vocab
    else:
        pos_vocab = dict()
    for sentence in ud_parses:
        for token in sentence:
            if str(token) in pos_vocab.keys():
                pos_tags.append(pos_vocab[str(token)])
            else:
                i = random.sample(range(17), 1)
                pos_vocab[str(token)] = i[0]
                pos_tags.append(i[0])

    pos_tags = torch.as_tensor(pos_tags)
  
    return pos_tags, pos_vocab

def create_data(filename: str, lm, w2i, pos_vocab=None):
    ud_parses = parse_corpus(filename)
    
    sen_reps = fetch_sen_reps_lstm(ud_parses, lm, w2i)
    pos_tags, pos_vocab = fetch_pos_tags(ud_parses, pos_vocab=pos_vocab)
    
    return sen_reps, pos_tags, pos_vocab

def create_pos_tags_control(filename: str, pos_vocab=None):
    ud_parses = parse_corpus(filename)
    
    pos_tags, pos_vocab = fetch_pos_tags_control(ud_parses, pos_vocab=pos_vocab)
    
    return pos_tags, pos_vocab


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(650,17)

    def forward(self, x):
        x = self.linear(x)
        x = torch.flatten(x, 1)
        #x = F.softmax(x)
        return x

def pos_probe_linear(train_x, train_y, dev_x, dev_y, control=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 10e-4
    batch_size = 256
    epochs = 10000
    classifier = LinearClassifier()
    classifier = classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    all_loss = []
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    dev_x = dev_x.to(device)
    dev_y = dev_y.to(device)

    accuracy = torchmetrics.Accuracy()
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            x_batch, y_batch = train_x[i:i+batch_size], train_y[i:i+batch_size]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = classifier(x_batch)

            loss = criterion(output, y_batch)
            all_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1000 == 0 :
            print(epoch)

    if control:
        torch.save(classifier, "linear_POS_LSTM_control.pt")
    else:
        torch.save(classifier, "linear_POS_LSTM.pt")
        
    return classifier

class nonLinearClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(650,512)
    self.linear2 = nn.Linear(512,17)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.linear1(x)
    x = self.sigmoid(x)
    x = self.linear2(x)
    x = torch.flatten(x, 1)
    #x = F.softmax(x)
    return x
    

def pos_probe_non_linear(train_x, train_y, dev_x, dev_y, control=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    non_linear_classifier = nonLinearClassifier()
    non_linear_classifier = non_linear_classifier.to(device)
    lr = 10e-4
    batch_size = 256
    epochs = 10000

    #doubling the criterions
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(non_linear_classifier.parameters(), lr=lr)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    dev_x = dev_x.to(device)
    dev_y = dev_y.to(device)

    all_loss = []

    accuracy = torchmetrics.Accuracy()
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            x_batch, y_batch = train_x[i:i+batch_size], train_y[i:i+batch_size]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = non_linear_classifier(x_batch)

            loss = criterion(output, y_batch)
            #print(loss)
            all_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1000 == 0 :
            print(epoch)
        
    if control:
        torch.save(non_linear_classifier, "non_linear_POS_LSTM_control.pt")
    else:
        torch.save(non_linear_classifier, "non_linear_POS_LSTM.pt")
    
    return non_linear_classifier

if __name__ == '__main__':
    lm = lstm  # or `lstm`
    w2i = vocab  # or `vocab`
    use_sample = False

    train_x, train_y, train_vocab = create_data(
        os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-train.conllu'),
        lm, 
        w2i
    )   

    dev_x, dev_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-dev.conllu'),
        lm, 
        w2i,
        pos_vocab=train_vocab
    )

    test_x, test_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-test.conllu'),
        lm,
        w2i,
        pos_vocab=train_vocab
    )

    train_y_control, train_vocab_control = create_pos_tags_control(
        os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-train.conllu'),
    )

    dev_y_control, _ = create_pos_tags_control(
        os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-dev.conllu'),
        pos_vocab=train_vocab_control
    )   

    test_y_control, _ = create_pos_tags_control(
        os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-test.conllu'),
        pos_vocab=train_vocab_control
    )
    classifier = pos_probe_linear(train_x, train_y, dev_x, dev_y)
    # classifier.eval()
    # output_test = classifier(test_x)
    # test_acc = accuracy(output_test, test_y)
    # print("The test accuracy is",test_acc.item())
    non_linear_classifier = pos_probe_non_linear(train_x, train_y, dev_x, dev_y)
    # non_linear_classifier.eval()
    # output_test = non_linear_classifier(test_x)
    # test_acc = accuracy(output_test, test_y)
    # print("The test accuracy fir the non_linear classifier is",test_acc.item())
    classifier_control = pos_probe_linear(train_x, train_y_control, dev_x, dev_y_control, True)
    non_linear_classifier_control = pos_probe_non_linear(train_x, train_y_control, dev_x, dev_y_control, True)