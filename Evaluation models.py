"loading models"
import torch
from tqdm import tqdm
from typing import List
from conllu import parse_incr, TokenList
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

    return FancyTree(f"{newick_str};")

_test_data_gpt = init_corpus_gpt(os.path.join('', 'data/en_ewt-ud-test.conllu'))
probe_gpt = torch.load("Tree_GPT_local.pt", map_location=torch.device('cpu'))
probe_gpt.eval()

test_loss, test_uuas_gpt = evaluate_probe(probe_gpt, _test_data_gpt)


print("The uuas score for the gpt model is", test_uuas_gpt)
