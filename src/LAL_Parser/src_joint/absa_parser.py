import os
import sys
sys.path.append(r'./LAL_Parser/src_joint')
import uuid
import torch
import numpy as np
import KM_parser
tokens = KM_parser
import nltk
# nltk.download('/home/yaolong/Dual-GNN-sentiment/LAL-Parser/averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, sent_tokenize

uid = uuid.uuid4().hex[:6]

REVERSE_TOKEN_MAPPING = dict([(value, key) for key, value in tokens.BERT_TOKEN_MAPPING.items()])

def torch_load1(load_path):
    if KM_parser.use_cuda:
        return torch.load(load_path, map_location=KM_parser.device)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)
    
    
class Config(object):
    model_path_base = '/home/yaolong/Dual-GNN-sentiment/LAL-Parser/best_model/best_parser.pt'
    contributions = 0


class ParseHead(object):
    def __init__(self, config) -> None:

        print("Loading model from {}...".format(config.model_path_base))
        assert config.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

        info = torch_load1(config.model_path_base)
        # info = torch.load(r'/home/yaolong/Dual-GNN-sentiment/LAL-Parser/best_model/best_parser.pt', map_location=KM_parser.device)
        assert 'hparams' in info['spec'], "Older savefiles not supported"
        self.parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
        self.parser.contributions = (config.contributions == 1)

    def parse_heads(self, sentence):
        self.parser.eval()
        with torch.no_grad():
            sentence = sentence.strip()
            split_mothod = lambda x: x.split(' ')
            tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(tag, tag), REVERSE_TOKEN_MAPPING.get(word, word)) for word, tag in nltk.pos_tag(split_mothod(sentence))]]
            syntree, _, arc = self.parser.parse_batch(tagged_sentences)  # 返回句法树 (syntree)、额外信息（未用到）、以及依存关系弧矩阵 (arc)
            arc_np = np.asarray(arc, dtype='float32')  # 将 arc 转换为 NumPy 数组

        return arc_np, syntree


config = Config()
headparser = ParseHead(config)