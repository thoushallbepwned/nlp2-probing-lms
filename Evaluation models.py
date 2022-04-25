"loading models"
import torch
from tqdm import tqdm
from typing import List
from conllu import parse_incr, TokenList
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ete3 import Tree as EteTree
from ete3 import Tree
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import optim
from collections import defaultdict
from lstm.model import RNNModel
import random
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# If stuff like `: str` and `-> ..` seems scary, fear not!
# These are type hints that help you to understand what kind of argument and output is expected.
def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))

    return ud_parses
ud_parses = parse_corpus("data/en_ewt-ud-train.conllu")

def fetch_sen_reps_tree(ud_parses: List[TokenList], model, tokenizer, concat=False):
    print("fetching sentence representations, might take a while...")
    print("")
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
            concat_dict[j] = [j + i for i in range(len_i)]
            j += len_i
            token_list += token_return
        token_list = torch.LongTensor(token_list)
        model.eval()
        with torch.no_grad():
            out_sentence = model(input_ids=token_list, output_hidden_states=True)
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

def evaluate_probe(probe, _data):
    probe.eval()
    x, y = _data
    loss_function = L1DistanceLoss()
    loss_function.eval()
    uuas_list = []

    with torch.no_grad():
        output = probe(x)
        length_batch = torch.count_nonzero(x, dim=1)[:, 0]
        loss_score, _ = loss_function(output, y, length_batch)
        for i in range(output.shape[0]):
            uuas_list.append(calc_uuas(output[i, :, :], y[i, :, :]))
        uuas_score = sum(uuas_list) / len(uuas_list)

    return loss_score, uuas_score


from ete3 import Tree as EteTree


def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()
    mst = minimum_spanning_tree(distances).toarray()
    mst[mst > 0] = 1.

    return mst


def edges(mst):
    edges = set()
    n_nodes = mst.shape[0]

    for i in range(n_nodes):
        for j in range(n_nodes):
            if mst[i][j] != 0:
                edges.add((i + 1, j + 1))

    return edges


def calc_uuas(pred_distances, gold_distances):
    gold_distances = gold_distances[gold_distances[0, :] != -1]
    valid_cols = [col_idx for col_idx, col in enumerate(torch.split(gold_distances, 1, dim=1)) if
                  not torch.all(col == -1)]
    gold_distances = gold_distances[:, valid_cols]
    sen_len = gold_distances.shape[0]
    pred_distances = pred_distances[:sen_len, :sen_len]
    gold_mst = create_mst(gold_distances)
    pred_mst = create_mst(pred_distances)
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    pred_in_gold = len(pred_edges.intersection(gold_edges))
    uuas = pred_in_gold / len(gold_distances)

    return uuas

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


import torch.nn as nn
import torch


class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim

        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))

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
        transposed = transformed.transpose(1, 2)

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
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=(1, 2))
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents

        else:
            batch_loss = torch.tensor(0.0)

        return batch_loss, total_sents
def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)

    return FancyTree(f"{newick_str};")

    return FancyTree(f"{newick_str};")

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
    torch.save(embs, "tree_rep_control.pt")
    torch.save(gold_distances, "tree_rep_control_distance.pt")

    return embs, gold_distances


#_test_data_gpt = init_corpus_gpt(os.path.join('', 'data/en_ewt-ud-test.conllu'))
_test_data_gpt = init_corpus_control_gpt(os.path.join('', 'data/en_ewt-ud-test.conllu'))

torch.save(_test_data_gpt, "test_data_gpt_control.pt")

def evaluation(data, model):
    test_data = torch.load(data)
    probe = torch.load(model, map_location = torch.device('cpu'))
    probe.eval()
    test_loss, test_uuas =evaluate_probe(probe, test_data)
    print("Currently evaluating:", model)
    print("")
    print("the uuas score for the model is", test_uuas)
    print("The test loss for the model is", test_loss)
    return



evaluation('test_data_gpt.pt', "Tree_GPT_local_cpu.pt")
evaluation('test_data_gpt_control.pt', "Tree_GPT_control_local_cpu.pt")

#_test_data_gpt = torch.load('test_data_gpt.pt')
#probe_gpt = torch.load("Tree_GPT_local.pt", map_location=torch.device('cpu'))
#probe_gpt.eval()

#test_loss, test_uuas_gpt = evaluate_probe(probe_gpt, _test_data_gpt)


#print("The uuas score for the gpt model is", test_uuas_gpt)
