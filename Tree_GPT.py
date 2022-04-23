from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
import nltk
from ete3 import Tree as EteTree
from ete3 import Tree
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import optim

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')


# If stuff like `: str` and `-> ..` seems scary, fear not! 
# These are type hints that help you to understand what kind of argument and output is expected.
def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

def rec_tokentree_to_nltk(tokentree):
    token = tokentree.token["form"]
    tree_str = f"({token} {' '.join(rec_tokentree_to_nltk(t) for t in tokentree.children)})"

    return tree_str


def tokentree_to_nltk(tokentree):
    from nltk import Tree as NLTKTree

    tree_str = rec_tokentree_to_nltk(tokentree)

    return NLTKTree.fromstring(tree_str)

class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)
        
    def __str__(self):
        return self.get_ascii(show_internal=True)
    
    def __repr__(self):
        return str(self)


def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx
    
def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)

    return FancyTree(f"{newick_str};")

def parse_corpus(filename):
    from conllu import parse_incr

    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))
    
    return ud_parses

def create_gold_distances(corpus):
    all_distances = []
    sen_length = [len(i) for i in corpus]
    max_length = max(sen_length)

    for item in (corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        distances = torch.full((max_length, max_length), -1)

        for i in range(sen_len):
            for j in range(sen_len):
                node_i = ete_tree&f"{i+1}"
                node_j = ete_tree&f"{j+1}"
                distances[i][j] = node_i.get_distance(node_j)

        all_distances.append(distances)

    return all_distances

def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.
    
    return mst

def edges(mst):
    edges = set()
    n_nodes = mst.shape[0]

    for i in range(n_nodes):
        for j in range(n_nodes):
            if mst[i][j] != 0:
                edges.add((i+1, j+1))

    return edges

def calc_uuas(pred_distances, gold_distances):
    
    gold_distances = gold_distances[gold_distances[0,:] != -1]
    valid_cols = [col_idx for col_idx, col in enumerate(torch.split(gold_distances, 1, dim=1)) if not torch.all(col == -1)]
    gold_distances = gold_distances[:, valid_cols]
    sen_len = gold_distances.shape[0]
    pred_distances = pred_distances[:sen_len,:sen_len]
    gold_mst = create_mst(gold_distances)
    pred_mst = create_mst(pred_distances)
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    pred_in_gold = len(pred_edges.intersection(gold_edges))
    uuas = pred_in_gold/len(gold_distances)
    
    return uuas

class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """
    def __init__(self, model_dim, rank, device):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim
        
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))
        
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        
        batchlen, seqlen, rank = transformed.size()
        
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)
        
        diffs = transformed - transposed
        
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)

        return squared_distances

    
class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()

        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=(1,2))
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        
        else:
            batch_loss = torch.tensor(0.0)
        
        return batch_loss, total_sents

def fetch_sen_reps_tree(ud_parses: List[TokenList], model, tokenizer, concat=False) -> Tensor:
    representation_size = 768
    out = []
    sen_length = [len(i) for i in ud_parses]
    max_length = max(sen_length)
    
    for sentence in tqdm(ud_parses):
        j = 0
        concat_dict = {}
        token_list = []
        space_after = False
        for token in sentence:
            
            test_var = token["misc"]
            
            if space_after:
                token_return = tokenizer.encode(str(token), add_prefix_space=True)
            else:
                token_return = tokenizer.encode(str(token))
            if test_var:
                space_after = False
            else:
                space_after = True        
            
            len_i = len(token_return)
            concat_dict[j] = [j+i for i in range(len_i)]
            j += len_i
            token_list += token_return
        token_list = torch.LongTensor(token_list)
        model.eval()
        with torch.no_grad():          
            out_sentence = model(input_ids = token_list, output_hidden_states=True)
        out_sentence = out_sentence["hidden_states"][-1].squeeze()        
        
        out_sent = torch.zeros(max_length, representation_size)
        for i, key in enumerate(concat_dict):
            out_sent[i] = torch.mean(out_sentence[concat_dict[key]], axis=0)
        out.append(out_sent)
    
    out = torch.stack(out)
    return out

def init_corpus_gpt(path, concat=False, cutoff=None):
    """ Initialises the data of a corpus.
    
    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating 
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]

    embs = fetch_sen_reps_tree(corpus, model, tokenizer, concat=concat)    
    gold_distances = torch.stack(create_gold_distances(corpus))
    
    return embs, gold_distances

def evaluate_probe(probe, _data):
    probe.eval()
    x, y = _data
    loss_function =  L1DistanceLoss()
    loss_function.eval()
    uuas_list = []
    
    with torch.no_grad():
        output = probe(x)
        length_batch = torch.count_nonzero(x, dim=1)[:,0]
        loss_score, _ = loss_function(output, y, length_batch)
        for i in range(output.shape[0]):
            uuas_list.append(calc_uuas(output[i,:,:], y[i,:,:]))
        uuas_score = sum(uuas_list)/len(uuas_list)
    
    return loss_score, uuas_score


# Feel free to alter the signature of this method.
def train(_data, control=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_dim = 768
    rank = 64
    lr = 10e-4
    batch_size = 24
    epochs = 200

    probe = StructuralProbe(emb_dim, rank, device=device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()
    x, y = _data
    x = x.to(device)
    y = y.to(device)
    dev_losses = []
    dev_uuass = []

    for epoch in tqdm(range(epochs)):

        for i in range(0, len(corpus), batch_size):
            x_batch, y_batch = x[i:i+batch_size], y[i:i+batch_size]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            output = probe(x_batch)
            length_batch = torch.count_nonzero(x_batch, dim=1)
            batch_loss, _ = loss_function(output, y_batch, length_batch[:,0])

            batch_loss.backward()
            optimizer.step()
        
        # dev_loss, dev_uuas = evaluate_probe(probe, _dev_data)
        # dev_losses.append(dev_loss)
        # dev_uuass.append(dev_uuas)

        # Using a scheduler is up to you, and might require some hyper param fine-tuning
        #scheduler.step(dev_loss)  
    
        if epoch % 20 == 0 :
            print(epoch)
            # model.eval()

            # print(f"Validation loss on batch {epoch}: {dev_loss}")
            # print(f"UUA on batch {epoch}: {dev_uuas}")
        
    # test_loss, test_uuas = evaluate_probe(probe, _test_data)
    if control:
        torch.save(probe, "Tree_GPT_control.pt")  
    else:
        torch.save(probe, "Tree_GPT.pt")    
    return probe

def create_gold_distances_control(corpus):
    all_distances = []
    sen_length = [len(i) for i in corpus]
    max_length = max(sen_length)

    for item in (corpus):
        tokentree = item.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        root = int(ete_tree.name)
        ete_tree = Tree()
        A = ete_tree.add_child(name=str(0))
        B = ete_tree.add_child(name=str(sen_len-1))
        distances = torch.full((max_length, max_length), -1)

        for i in range(1,sen_len-1):
            if i != root:
                choice = random.sample(range(3),1)
                if choice[0] == 0:
                    A.add_child(name=str(i))
                if choice[0] == 1:
                    B.add_child(name=str(i))
                if choice[0] == 2:
                    ete_tree.add_child(name=str(i))

        for i in range(sen_len):
            for j in range(sen_len):
                if i != root:
                    node_i = ete_tree&f"{i}"
                else:
                    node_i = ete_tree
                if j != root:
                    node_j = ete_tree&f"{j}"
                else:
                    node_j = ete_tree
                distances[i][j] = node_i.get_distance(node_j)

        all_distances.append(distances)

    return all_distances

def init_corpus_control_gpt(path, concat=False, cutoff=None):
    """ Initialises the data of a corpus.
    
    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating 
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]

    embs = fetch_sen_reps_tree(corpus, model, tokenizer, concat=concat)    
    gold_distances = torch.stack(create_gold_distances_control(corpus))
    
    return embs, gold_distances

if __name__ == '__main__':
    corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')
    train_data = init_corpus_gpt(os.path.join('', 'data/en_ewt-ud-train.conllu'))
    train_data_control = init_corpus_control_gpt(os.path.join('', 'data/en_ewt-ud-train.conllu'))
    probe = train(train_data)
    probe_control = train(train_data_control, True)
