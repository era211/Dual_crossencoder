import os
import sys
import random
import logging
import argparse
import traceback
import itertools
from tqdm import tqdm, trange
# --- data process
import numpy as np
import pandas as pd
import pickle
import pickle as cPickle
import json
# ---
from scorer import *  # calculate coref metrics
# For clustering
from graphviz import Graph
import networkx as nx
# Record training process
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from transformers import RobertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

#  Dynamically adding module search paths
for pack in os.listdir("/root/autodl-tmp/Rationale4CDECR-main/src"):
    sys.path.append(os.path.join("src", pack))
sys.path.append("/root/autodl-tmp/Rationale4CDECR-main/src/shared/")
from classes import *  # make sure classes in "/src/shared/" can be imported.
from bcubed_scorer import *
from coarse import *
from fine_all import *

parser = argparse.ArgumentParser(description='Training a cross-encoder')
parser.add_argument('--config_path',
                    type=str,
                    default='/root/autodl-tmp/Rationale4CDECR-main/configs/main/ecb/baseline.json',
                    help=' The path configuration json file')

parser.add_argument('--out_dir',
                    type=str,
                    default='/root/autodl-tmp/Rationale4CDECR-main/outputs/main/ecb/cf/baseline/best_model',
                    help=' The directory to the output folder')
parser.add_argument('--out_dir_test',
                    type=str,
                    default='/root/autodl-tmp/Rationale4CDECR-main/outputs/main/ecb/cf/baseline/eval_results',
                    help=' The directory to the output folder')
parser.add_argument('--mode',
                    type=str,
                    default='train',
                    help='train or eval')

# parser.add_argument('--eval',
#                     dest='evaluate_dev',
#                     action='store_true',
#                     help='evaluate_dev')

parser.add_argument('--random_seed',
                    type=int,
                    default=5,
                    help=' Random Seed')
# GPU index
parser.add_argument('--gpu_num',
                    type=int,
                    default=1,
                    help=' A single GPU number')
parser.add_argument('--load_data', default=True, type=bool, help='load data')
parser.add_argument('--load_test_data', default=True, type=bool, help='load test data')
parser.add_argument('--save_data', default=False, type=bool, help='save data')
parser.add_argument('--test', default=False, type=bool, help='test')
parser.add_argument('--alpha', default=0.15, type=float)
parser.add_argument('--beta', default=0.15, type=float)

# load all config parameters for training
args = parser.parse_args()
assert args.mode in ['train', 'eval'], "mode must be train or eval!"
# In the `train' mode, the best model and the evaluation results of the dev corpus are saved.
# In the `eval' mode, evaluation results of the test corpus are saved.

# 读取配置文件
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# out_dir
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# save basicConfig log
logging.basicConfig(filename=os.path.join(args.out_dir, "crossencoder_train_log.txt"),
                    level=logging.DEBUG,
                    filemode='w')
# save config into out_dir for easily checking
with open(os.path.join(args.out_dir, 'AD_crossencoder_train_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

# use cuda and set seed
seed = args.random_seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if args.gpu_num != -1 and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    args.use_cuda = True
    set_seed(seed)
    # device = torch.device("cuda:1" if args.use_cuda else "cpu")
    # 检查可见的 GPU 设备
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"Visible CUDA devices: {visible_devices}")
    print('Training with CUDA')
else:
    args.use_cuda = False

# Set global variables
best_score = None
patience = 0  # for early stopping
comparison_set = set()
tokenizer = RobertaTokenizer.from_pretrained("/root/autodl-tmp/PLM/roberta-base")  # tokenizer
# 这个tokenizer在哪里使用了：构建训练句子对的时候，对句子对进行token化
# use train/dev data in 'train' mode, and use test data in 'eval' mode
print('Loading fixed dev set')
# 这些数据对的产生过程
ecb_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/ecb/dev/dev_pairs')
fcc_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/fcc/dev/dev_pairs')
gvc_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/gvc/dev/dev_pairs')

print('Loading fixed test set')
ecb_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/ecb/test/test_pairs')
fcc_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/fcc/test/test_pairs')
gvc_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/gvc/test/test_pairs')
print('successful loading fixed dev & test sets')

nn_generated_fixed_eval_pairs = {
    'ecb':
        {
            'dev': ecb_dev_set,
            'test': ecb_test_set
        },
    'fcc':
        {
            'dev': fcc_dev_set,
            'test': fcc_test_set
        },
    'gvc':
        {
            'dev': gvc_dev_set,
            'test': gvc_test_set
        }
}


def get_sents(sentences, sentence_id, window=config_dict["window_size"]):
    ''' Get sentence instances S_{i-w},..., S_{i},..., S_{i+w}
        params:
            sentences: a list of `sentence' instances in a document
            sentence_id: the id of the `sentence' instance where the event mention occurs in the document
            window: the window size, it determines how many sentences around the event mention are included in the discourse
        return:
            (a list of `sentence' instances in the window,
                the offset of the mention sentence in the window) 返回当前mention对应句子前后各Windows个范围之内的句子（相当于包含了上下文），当前句子在窗口范围中对应位置
    '''
    lookback = max(0, sentence_id - window)  # 当前mention所在的句子的前窗口范围个句子
    lookforward = min(sentence_id + window, max(sentences.keys())) + 1  # 当前mention所在句子的后窗口范围个句子
    return ([sentences[_id]
             for _id in range(lookback, lookforward)], sentence_id - lookback)


def structure_pair(mention_1,
                   mention_2,
                   doc_dict,
                   window=config_dict["window_size"]):
    ''' Curate the necessary information for model input
        params:
            mention_1: the first mention instance
            mention_2: the second mention instance
            doc_dict: a dictionary of documents,
                where the key is the document id and the value is the document instance
            window: the window size, it determines how many sentences around
                the event mention are included in the discourse
        return:
            record: it the necessary information for model input
    '''
    try:
        sents_1, sent_id_1 = get_sents(doc_dict[mention_1.doc_id].sentences,  # 得到当前mention所在句子窗口范围的上下文句子以及在+-窗口范围中的相对位置
                                       mention_1.sent_id,
                                       window)  # doc_dict[mention_1.doc_id].sentences得到当前mention对应的doc_id文档中的所有句子，mention_1.sent_id得到当前mention对应句子的索引
        sents_2, sent_id_2 = get_sents(doc_dict[mention_2.doc_id].sentences,
                                       mention_2.sent_id, window)
        tokens, token_map, offset_1, offset_2 = tokenize_and_map_pair(
            # tokens是mention上下文窗口中所有原始句子编码后的token数字序列，token_map是原始句子中每个单词索引和编码的各个token索引之间的映射，其他两个是mention所在句子在token序列中的对应位置
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer)
        start_piece_1 = token_map[offset_1 + mention_1.start_offset][0]  # 得到mention在token序列中的起始位置
        if offset_1 + mention_1.end_offset in token_map:
            end_piece_1 = token_map[offset_1 + mention_1.end_offset][-1]  # 得到mention的token表示的起始和结束索引
        else:
            end_piece_1 = token_map[offset_1 + mention_1.start_offset][-1]
        start_piece_2 = token_map[offset_2 + mention_2.start_offset][0]
        if offset_2 + mention_2.end_offset in token_map:
            end_piece_2 = token_map[offset_2 + mention_2.end_offset][-1]
        else:
            end_piece_2 = token_map[offset_2 + mention_2.start_offset][-1]
        label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]  # 通过gold_tag来判断两个mention是否共指
        record = {
            "sentence": tokens,
            # the embedding of pairwise mention data, i.e., tokenizer(sent1_with_discourse, sent2_with_discourse)
            "label": label,  # coref (1.0) or non-coref (0.0)
            "start_piece_1": [start_piece_1],  # the start and end offset of trigger_1 pieces in "sentence"
            "end_piece_1": [end_piece_1],
            "start_piece_2": [start_piece_2],  # # the start and end offset of trigger_2 pieces in "sentence"
            "end_piece_2": [end_piece_2]
        }
    except:
        if window > 0:
            return structure_pair(mention_1, mention_2, doc_dict, window - 1)
        else:
            traceback.print_exc()
            sys.exit()
    return record


def structure_dataset_for_eval(data_set,
                               eval_set='dev'):
    if not args.load_data:
        assert eval_set in ['dev', 'test'], "please check the eval_set!"
        processed_dataset = []
        doc_dict = {
            key: document
            for topic in data_set.topics.values()
            for key, document in topic.docs.items()
        }  # data_set用来构建原始数据集中所有主题的文档字典
        train_set_name = config_dict["training_dataset"]
        test_set_name = config_dict["test_dataset"]
        if eval_set == 'dev':
            # even in ood test, dev and train set are from the same corpus.
            pairs = nn_generated_fixed_eval_pairs[train_set_name][eval_set]
        elif eval_set == 'test':
            pairs = nn_generated_fixed_eval_pairs[test_set_name][
                eval_set]  # 字典类型 /retrieved_data/main/ecb/test/test_pairs'  这个数据集中保存的都是mention对
        pairs = list(pairs)  # 将字典类型转换为列表类型
        for mention_1, mention_2 in pairs:  # 从提及对数据列表中分别读取每一对mention
            record = structure_pair(mention_1, mention_2,
                                    doc_dict)  # 得到两个mention的上下文句子的token序列，标签值，mention分别在它们对应句子的token序列中的起始和结束位置
            processed_dataset.append(record)
        sentences = torch.tensor(
            [record["sentence"] for record in processed_dataset])
        labels = torch.tensor([record["label"] for record in processed_dataset])
        start_pieces_1 = torch.tensor(
            [record["start_piece_1"] for record in processed_dataset])
        end_pieces_1 = torch.tensor(
            [record["end_piece_1"] for record in processed_dataset])
        start_pieces_2 = torch.tensor(
            [record["start_piece_2"] for record in processed_dataset])
        end_pieces_2 = torch.tensor(
            [record["end_piece_2"] for record in processed_dataset])
        print(labels.sum() / float(labels.shape[0]))
        tensor_dataset = TensorDataset(sentences, start_pieces_1, end_pieces_1, \
                                       start_pieces_2, end_pieces_2, labels)

        if args.save_data:
            # 将数据打包进一个字典
            data_dict = {
                'tensor_dataset': tensor_dataset,
                'pairs': pairs,
                'doc_dict': doc_dict
            }

            # 保存字典到 .pkl 文件
            file_path = '/root/autodl-tmp/Rationale4CDECR-main/data_preparation/cf/dev_data.pkl'  # 文件路径
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print("数据已成功保存到文件。")
    else:
        if eval_set == 'dev':
            print('加载dev数据...')
            # 指定文件路径
            file_path = '/root/autodl-tmp/Rationale4CDECR-main/data_preparation/cf/dev_data.pkl'
        elif eval_set == 'test':
            print('加载test数据...')
            # 指定文件路径
            file_path = '/root/autodl-tmp/Rationale4CDECR-main/data_preparation/cf/test_data.pkl'

        # 打开文件并加载数据
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 现在可以从字典中获取各个数据
        if eval_set == 'dev':
            tensor_dataset = data_dict['dev_tensor_dataset']
        elif eval_set == 'test':
            tensor_dataset = data_dict['tensor_dataset']
        pairs = data_dict['pairs']
        doc_dict = data_dict['doc_dict']

        print("数据已成功从文件中读取...")
    return tensor_dataset, pairs, doc_dict


def structure_data_for_train(df):  # 构建训练数据对
    if not args.load_data:
        max_seq_length = 512
        all_data_index = df.index
        all_labels, all_sentences = [], []
        all_start_piece_1, all_end_piece_1 = [], []
        all_start_piece_2, all_end_piece_2 = [], []
        for ID in all_data_index:  # 处理每一行数据
            # get 'label'
            label = [float(df['label'][ID])]
            # get 'sentence'
            sentences_text_1 = df['text_1'][ID]
            sentences_text_2 = df['text_2'][ID]
            text_1_length = len(sentences_text_1.split(' '))
            embeddings = tokenizer(sentences_text_1,
                                   sentences_text_2,
                                   max_length=max_seq_length,
                                   truncation=True,
                                   padding="max_length")["input_ids"]
            # get start/end_piece_1/2
            counter = 0
            new_tokens = tokenizer.convert_ids_to_tokens(embeddings)  # 将编码数字转换为对应的token
            total_tokens_num = df['total_tokens_num'][ID]
            token_map = dict(list(map(lambda x: (x, []), np.arange(
                total_tokens_num))))  # 以第一句为例，句子1和句子2的总长度为173，tokenizer后得到新的token，包括原本词和子词，所以这里将新产生的token分别与原来的173个词对应起来
            for i, token in enumerate(new_tokens):
                if ((i + 1) < len(new_tokens) - 1) and (new_tokens[i] == "</s>") and (new_tokens[i + 1] == "</s>"):
                    counter = text_1_length - 1
                else:
                    pass
                if token == "<s>" or token == "</s>" or token == "<pad>":
                    continue
                elif token[0] == "Ġ" or new_tokens[i - 1] == "</s>":
                    counter += 1
                    token_map[counter].append(i)
                else:
                    token_map[counter].append(i)
                    continue
            trigger_1_abs_start = df['trigger_1_abs_start'][ID]
            trigger_1_abs_end = df['trigger_1_abs_end'][ID]
            trigger_2_abs_start = df['trigger_2_abs_start'][ID]
            trigger_2_abs_end = df['trigger_2_abs_end'][ID]
            ##get start/end_piece_1
            start_piece_1 = token_map[trigger_1_abs_start][0]  # 得到新token后的触发词的索引
            if trigger_1_abs_end in token_map:
                end_piece_1 = token_map[trigger_1_abs_end][-1]
            else:
                end_piece_1 = token_map[trigger_1_abs_start][-1]
            start_piece_2 = token_map[trigger_2_abs_start][0]
            if trigger_2_abs_end in token_map:
                end_piece_2 = token_map[trigger_2_abs_end][-1]
            else:
                end_piece_2 = token_map[trigger_2_abs_start][-1]
            all_sentences.append(embeddings)
            all_start_piece_1.append([start_piece_1])
            all_end_piece_1.append([end_piece_1])
            all_start_piece_2.append([start_piece_2])
            all_end_piece_2.append([end_piece_2])
            all_labels.append(label)
        data_set_in_tensor = TensorDataset(torch.tensor(all_sentences), torch.tensor(all_start_piece_1), \
                                           torch.tensor(all_end_piece_1), torch.tensor(all_start_piece_2), \
                                           torch.tensor(all_end_piece_2), torch.tensor(all_labels))

        if args.save_data:
            # 将数据打包进一个字典
            data_dict = {
                'data_set_in_tensor': data_set_in_tensor
            }

            # 保存字典到 .pkl 文件
            file_path = '/root/autodl-tmp/Rationale4CDECR-main/data_preparation/cf/train_data.pkl'  # 文件路径
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print("数据已成功保存到文件。")
    else:
        print('加载数据...')
        # 指定文件路径
        file_path = '/root/autodl-tmp/Rationale4CDECR-main/data_preparation/cf/train_data.pkl'

        # 打开文件并加载数据
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 现在可以从字典中获取各个数据
        data_set_in_tensor = data_dict['data_set_in_tensor']

        print("数据已成功从文件中读取...")
    return data_set_in_tensor


def collate_fn(batch):
    all_sentences, all_start_piece_1, all_end_piece_1, all_start_piece_2, all_end_piece_2, all_labels = zip(*batch)

    # 获取RoBERTa的mask token id和padding token id
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    pad_token_id = tokenizer.pad_token_id

    # 处理每个句子的编码，将指定范围之外的部分替换为mask token，但保留padding部分
    c_only_sentences = []
    e_only_sentences = []
    for i in range(len(all_sentences)):
        sentence = all_sentences[i]
        start_idx_1 = all_start_piece_1[i].item()
        end_idx_1 = all_end_piece_1[i].item()
        start_idx_2 = all_start_piece_2[i].item()
        end_idx_2 = all_end_piece_2[i].item()

        # 创建mask后的句子
        c_masked_sentence = sentence.clone()  # 深拷贝原句子
        e_masked_sentence = sentence.clone()  # 深拷贝原句子
        # Mask除指定范围以外的部分，但保留padding
        for j in range(len(e_masked_sentence)):
            if e_masked_sentence[j] != pad_token_id:
                if not (start_idx_1 <= j <= end_idx_1 or start_idx_2 <= j <= end_idx_2):
                    e_masked_sentence[j] = mask_token_id

        # 遍历每个token，将 start_piece_1 ~ end_piece_1 和 start_piece_2 ~ end_piece_2 范围内的部分 mask 掉
        for j in range(len(c_masked_sentence)):
            if c_masked_sentence[j] != pad_token_id:  # 跳过 padding 部分
                if start_idx_1 <= j <= end_idx_1 or start_idx_2 <= j <= end_idx_2:
                    c_masked_sentence[j] = mask_token_id  # 将这些范围内的token替换为mask token

        # 将处理后的句子添加到列表
        e_only_sentences.append(e_masked_sentence)

        c_only_sentences.append(c_masked_sentence)

    # 将处理后的句子构建成批次
    c_sentences = torch.stack(c_only_sentences)
    e_sentences = torch.stack(e_only_sentences)
    # 将其他部分处理为tensor，并返回批次数据
    sentences = torch.stack(all_sentences)
    start_pieces_1 = torch.tensor(all_start_piece_1).reshape(-1, 1)
    end_pieces_1 = torch.tensor(all_end_piece_1).reshape(-1, 1)
    start_pieces_2 = torch.tensor(all_start_piece_2).reshape(-1, 1)
    end_pieces_2 = torch.tensor(all_end_piece_2).reshape(-1, 1)
    labels = torch.tensor(all_labels).reshape(-1, 1)

    return c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels


def get_optimizer(model):
    '''
       define the optimizer
    '''
    lr = config_dict["lr"]
    optimizer = None
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config_dict["optimizer"] == 'adadelta':
        optimizer = optim.Adadelta(parameters,
                                   lr=lr,
                                   weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'adam':
        optimizer = optim.Adam(parameters,
                               lr=lr,
                               weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'sgd':
        optimizer = optim.SGD(parameters,
                              lr=lr,
                              momentum=config_dict["momentum"],
                              nesterov=True)
    assert (optimizer is not None), "Config error, check the optimizer field"
    return optimizer


def get_scheduler(optimizer, len_train_data):
    ''' linear learning rate scheduler
            params: optimizer
            len_train_data: total number of training data
    '''
    batch_size = config_dict["accumulated_batch_size"]
    epochs = config_dict["epochs"]
    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * config_dict["warmup_proportion"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                                num_train_steps)
    return scheduler


# For mention clustering
def find_cluster_key(node, clusters):
    if node in clusters:
        return node
    for key, value in clusters.items():
        if node in value:
            return key
    return None


def is_cluster_merge(cluster_1, cluster_2, mentions, model, doc_dict):
    if config_dict["oracle"]:
        return True
    score = 0.0
    sample_size = 100
    global comparison_set
    if len(cluster_1) > sample_size:
        c_1 = random.sample(cluster_1, sample_size)
    else:
        c_1 = cluster_1
    if len(cluster_2) > sample_size:
        c_2 = random.sample(cluster_2, sample_size)
    else:
        c_2 = cluster_2
    for mention_id_1 in c_1:
        records = []
        mention_1 = mentions[mention_id_1]
        for mention_id_2 in c_2:
            comparison_set = comparison_set | {frozenset([mention_id_1, mention_id_2])}
            mention_2 = mentions[mention_id_2]
            record = structure_pair(mention_1, mention_2, doc_dict)
            records.append(record)
        sentences = torch.tensor([record["sentence"] for record in records])
        labels = torch.tensor([record["label"] for record in records])
        start_pieces_1 = torch.tensor([record["start_piece_1"] for record in records])
        end_pieces_1 = torch.tensor([record["end_piece_1"] for record in records])
        start_pieces_2 = torch.tensor([record["start_piece_2"] for record in records])
        end_pieces_2 = torch.tensor([record["end_piece_2"] for record in records])
        # 将这些数据组合成一个 batch（注意使用 zip 打包）
        batch = list(zip(sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels))
        c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = collate_fn(
            batch)
        # 将数据移动到 GPU 上
        c_sentences = c_sentences.to(model.device)
        e_sentences = e_sentences.to(model.device)
        sentences = sentences.to(model.device)
        start_pieces_1 = start_pieces_1.to(model.device)
        end_pieces_1 = end_pieces_1.to(model.device)
        start_pieces_2 = start_pieces_2.to(model.device)
        end_pieces_2 = end_pieces_2.to(model.device)
        labels = labels.to(model.device)
        with torch.no_grad():
            out_dict = model(c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1,
                             start_pieces_2, end_pieces_2, labels)
            mean_prob = torch.mean(out_dict["probabilities"]).item()
            score += mean_prob
    return (score / len(cluster_1)) >= 0.5


def transitive_closure_merge(edges, mentions, model, doc_dict, graph,
                             graph_render):  # 对共指提及进行聚类
    clusters = {}
    inv_clusters = {}
    mentions = {mention.mention_id: mention for mention in mentions}
    print("\n@@@@@@@@合并聚类@@@@@@@@\n")
    for edge in tqdm(edges):  # prob大于0.5的两个提及的索引，标签和概率  [()]
        cluster_key = find_cluster_key(edge[0], clusters)
        alt_key = find_cluster_key(edge[1], clusters)
        if cluster_key == None and alt_key == None:
            cluster_key = edge[0]
            clusters[cluster_key] = set()
        elif cluster_key == None and alt_key != None:
            cluster_key = alt_key
            alt_key = None
        elif cluster_key == alt_key:
            alt_key = None
        # If alt_key exists, merge clusters
        perform_merge = True
        if alt_key:
            perform_merge = is_cluster_merge(clusters[cluster_key],
                                             clusters[alt_key], mentions,
                                             model, doc_dict)
        elif clusters[cluster_key] != set():
            new_elements = set([edge[0], edge[1]]) - clusters[cluster_key]
            if len(new_elements) > 0:
                perform_merge = is_cluster_merge(clusters[cluster_key],
                                                 new_elements, mentions, model,
                                                 doc_dict)
        if alt_key and perform_merge:
            clusters[cluster_key] = clusters[cluster_key] | clusters[alt_key]
            for node in clusters[alt_key]:
                inv_clusters[node] = cluster_key
            del clusters[alt_key]
        if perform_merge:
            if not (graph.has_edge(edge[0], edge[1])
                    or graph.has_edge(edge[1], edge[0])):
                graph.add_edge(edge[0], edge[1])  # 在图中给两个prob大于0.5的mention对添加边
                color = 'black'
                if edge[2] != 1.0:
                    color = 'red'
                graph_render.edge(edge[0],  # 给图dot添加边和颜色信息以及标签预测值
                                  edge[1],
                                  color=color,
                                  label=str(edge[3]))
            cluster = clusters[cluster_key]
            cluster.add(edge[0])
            cluster.add(edge[1])
            inv_clusters[edge[0]] = cluster_key
            inv_clusters[edge[1]] = cluster_key
    print(len(comparison_set))
    return clusters, inv_clusters


# eval the cross-encoder
def evaluate(model, encoder_model, dev_dataloader, dev_pairs, doc_dict,
             epoch_num):
    global best_score, comparison_set
    model = model.eval()
    offset = 0
    edges = set()
    saved_edges = []
    best_edges = {}
    mentions = set()
    acc_sum = 0.0
    all_probs = []
    for step, batch in enumerate(tqdm(dev_dataloader, desc="Test Batch")):
        batch = tuple(t.to(model.device) for t in batch)
        C_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
        if not config_dict["oracle"]:
            with torch.no_grad():
                out_dict = model(C_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1,
                                 start_pieces_2, end_pieces_2, labels)
        else:
            out_dict = {
                "accuracy": 1.0,
                "predictions": labels,
                "probabilities": labels
            }
        acc_sum += out_dict["accuracy"]
        predictions = out_dict["predictions"].detach().cpu().tolist()
        probs = out_dict["probabilities"].detach().cpu().tolist()
        for p_index in range(len(predictions)):  # batch中每一个样本的训练结果
            pair_0, pair_1 = dev_pairs[offset + p_index]  # 得到当前预测结果对应的mention对
            prediction = predictions[p_index]
            mentions.add(pair_0)
            mentions.add(pair_1)
            comparison_set = comparison_set | {frozenset([pair_0.mention_id, pair_1.mention_id])}
            if probs[p_index][0] > 0.5:
                if pair_0.mention_id not in best_edges or (probs[p_index][0] > best_edges[pair_0.mention_id][3]):
                    best_edges[pair_0.mention_id] = (pair_0.mention_id,
                                                     pair_1.mention_id,
                                                     labels[p_index][0],
                                                     probs[p_index][0])
                edges.add((pair_0.mention_id, pair_1.mention_id, labels[p_index][0], probs[p_index][0]))
            saved_edges.append((pair_0, pair_1, labels[p_index][0].detach().cpu().tolist(), probs[p_index][0]))
        offset += len(predictions)

    tqdm.write("Pairwise Accuracy: {:.6f}".format(acc_sum / float(len(dev_dataloader))))
    # writer.add_scalar('dev/pairwise_acc',acc_sum/float(len(dev_dataloader)),epoch_num)
    eval_edges(edges, mentions, model, doc_dict, saved_edges)  # edges保存prob大于0.5的mention对，saved_edges保存所有的mention对
    assert len(saved_edges) >= len(edges)
    return saved_edges


# eval the coref-metric based on edges among clusters
def eval_edges(edges, mentions, model, doc_dict, saved_edges):
    print(len(mentions))
    global best_score, patience
    dot = Graph(comment='Cross Doc Co-ref')
    G = nx.Graph()
    edges = sorted(edges, key=lambda x: -1 * x[3])  # 按照概率值prob对edges进行降序排序
    for mention in mentions:
        G.add_node(mention.mention_id)
        dot.node(mention.mention_id,
                 label=str((str(mention), doc_dict[mention.doc_id].sentences[
                     mention.sent_id].get_raw_sentence())))
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))
    # edges = [edge for edge in edges if edge not in bridges]
    clusters, inv_clusters = transitive_closure_merge(edges, mentions, model,
                                                      doc_dict, G, dot)  # 对共指提及进行聚类

    # Find Transitive Closure Clusters
    gold_sets = []
    model_sets = []
    ids = []
    model_map = {}
    gold_map = {}
    for mention in mentions:
        ids.append(mention.mention_id)
        gold_sets.append(mention.gold_tag)
        gold_map[mention.mention_id] = mention.gold_tag
        if mention.mention_id in inv_clusters:
            model_map[mention.mention_id] = inv_clusters[mention.mention_id]
            model_sets.append(inv_clusters[mention.mention_id])
        else:
            model_map[mention.mention_id] = mention.mention_id  # 不共指的，不在cluster中
            model_sets.append(mention.mention_id)
    model_clusters = [[thing[0] for thing in group[1]] for group in
                      itertools.groupby(sorted(zip(ids, model_sets), key=lambda x: x[1]), lambda x: x[1])]
    gold_clusters = [[thing[0] for thing in group[1]] for group in
                     itertools.groupby(sorted(zip(ids, gold_sets), key=lambda x: x[1]), lambda x: x[1])]
    if args.test:  # During Test
        print('saving gold_map, model_map...')
        # save the golden_map which groups all mentions based on the annotation.
        with open(os.path.join(args.out_dir_test, "gold_map"), 'wb') as f:
            pickle.dump(gold_map, f)
            # save the model_map which groups all mentions based on the coreference pipeline.
        with open(os.path.join(args.out_dir_test, "model_map"), 'wb') as f:
            pickle.dump(model_map, f)
        print('Saved!')
    else:
        pass
    # deprecated
    pn, pd = b_cubed(model_clusters, gold_map)
    rn, rd = b_cubed(gold_clusters, model_map)
    tqdm.write("Alternate = Recall: {:.6f} Precision: {:.6f}".format(pn/pd, rn/rd))
    p, r, f1 = bcubed(gold_sets, model_sets)
    tqdm.write("Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(p, r, f1))
    if best_score == None or f1 > best_score:
        tqdm.write("F1 Improved Saving Model")
        best_score = f1
        patience = 0
        if args.mode == 'train':  # During training
            # save the best model measured by b_cubded F1 in the current epoch
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "AD_crossencoder_best_model"),
            )
            # save the edges linking coreferential events on the dev corpus
            with open(os.path.join(args.out_dir, "crossencoder_dev_edges"), "wb") as f:
                cPickle.dump(saved_edges, f)
        else:  # During Test
            # save the edges linking coreferential events on the test corpus
            with open(os.path.join(args.out_dir_test, "crossencoder_test_edges"), "wb") as f:
                cPickle.dump(saved_edges, f)
            # dot.render(os.path.join(args.out_dir, "clustering"))
    else:
        patience += 1
        if patience > config_dict["early_stop_patience"]:
            print("Early Stopping")
            sys.exit()


def test_model(model, last_epoch=True):
    args.test = True
    args.mode = 'eval'
    if args.test:
        print("\n++==========test=========++\n")
        with open(config_dict["test_path"], 'rb') as f:  # test_data是从feature_sets中读取到的
            test_data = cPickle.load(f)
        topic_sizes = [
            len([
                mention for key, doc in topic.docs.items()
                for sent_id, sent in doc.get_sentences().items()
                for mention in sent.gold_event_mentions
            ]) for topic in test_data.topics.values()
        ]
        print(topic_sizes)
        print(sum(topic_sizes))
        print(sum([size * size for size in topic_sizes]))
        device = torch.device("cuda:1" if args.use_cuda else "cpu")
        event_encoder_path = config_dict['event_encoder_model']
        with open(event_encoder_path, 'rb') as f:
            params = torch.load(f, map_location=device)
            event_encoder = EncoderCosineRanker(device)
            event_encoder.load_state_dict(params)
            event_encoder = event_encoder.to(device).eval()
            event_encoder.requires_grad = False

        if last_epoch == True:
            if config_dict['eval_model_path'] == False:
                logging.info('Loading default model for eval...')
                eval_model_path = os.path.join(args.out_dir, 'AD_crossencoder_best_model')
            else:
                logging.info('Loading the specified model for eval...')
                eval_model_path = config_dict['eval_model_path']
            with open(eval_model_path, 'rb') as f:
                params = torch.load(f)
                model = CoreferenceCrossEncoder(device, args)
                model.load_state_dict(params)
                model = model.to(device).eval()
                model.requires_grad = False
        else:
            model = model
        test_event_pairs, test_pairs, test_docs = structure_dataset_for_eval(test_data,
                                                                             eval_set='test')  # test_event_pairs是编码后的上下文token数字序列以及标签，两个mention的起始和结束索引，test_pairs是mention对，test_docs是文档字典
        test_dataloader = DataLoader(test_event_pairs,
                                     sampler=SequentialSampler(test_event_pairs),  # 顺序采样
                                     collate_fn=collate_fn,
                                     batch_size=config_dict["batch_size"])
        evaluate(model, event_encoder, test_dataloader, test_pairs, test_docs, 0)

        from conll_eval.eval import result
        result()
        args.test = False
        args.mode = 'train'

def train_model(df, dev_set):
    device = torch.device("cuda:1" if args.use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))
    print(f"Using device: {device}")
    # load bi-encoder model
    event_encoder_path = config_dict['event_encoder_model']  # 用于事件编码
    with open(event_encoder_path, 'rb') as f:
        params = torch.load(f, map_location=device)  # 将读取到的数据加载到指定的设备上，默认为0卡
        event_encoder = EncoderCosineRanker(device)
        event_encoder.load_state_dict(params)
        event_encoder = event_encoder.to(device).eval()
        event_encoder.requires_grad = False
    model = CoreferenceCrossEncoder(device, args).to(device)
    train_data_num = len(df)  # baseline: 49864
    train_event_pairs = structure_data_for_train(df)  # 构建训练数据集，TensorDate类，返回值包括所有训练数据的embedding，触发词的开始和结束位置以及label
    dev_event_pairs, dev_pairs, dev_docs = structure_dataset_for_eval(dev_set, eval_set='dev')
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_data_num)
    train_dataloader = DataLoader(train_event_pairs,
                                  batch_size=config_dict["batch_size"],
                                  collate_fn=collate_fn,
                                  shuffle=True)
    dev_dataloader = DataLoader(dev_event_pairs,
                                sampler=SequentialSampler(dev_event_pairs),
                                collate_fn=collate_fn,
                                batch_size=config_dict["batch_size"])
    for epoch_idx in trange(int(config_dict["epochs"]),
                            desc="Epoch",
                            leave=True):
        print(f"\n===========epoch:{epoch_idx}===========")
        model = model.train()
        tr_loss = 0.0
        tr_p = 0.0
        tr_a = 0.0
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(device) for t in batch)
            c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
            # sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
            out_dict = model(c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1,
                             start_pieces_2, end_pieces_2, labels)
            loss = out_dict["loss"]
            precision = out_dict["precision"]
            accuracy = out_dict["accuracy"]
            loss.backward()
            tr_loss += loss.item()
            tr_p += precision.item()
            tr_a += accuracy.item()
            if ((step + 1) * config_dict["batch_size"]
            ) % config_dict["accumulated_batch_size"] == 0:
                # For main exp, we set batch_size as 10, accumulated_batch_size as 8.
                # This should be equivalent to the case of batch_size being 40 and accumulated_batch_size being 8
                batcher.set_description(
                    "Batch (average loss: {:.6f} precision: {:.6f} accuracy: {:.6f})"
                    .format(
                        tr_loss / float(step + 1),
                        tr_p / float(step + 1),
                        tr_a / float(step + 1),
                    ))
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config_dict["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        print(f"\n+++++++++++=dev:{epoch_idx}=+++++++++++")
        evaluate(model, event_encoder, dev_dataloader, dev_pairs, dev_docs,
                 epoch_idx)

        test_model(model, last_epoch=False)
    test_model(model, last_epoch=True)




def main():
    if args.mode == 'train':
        print("模型训练！")
        logging.info('Loading training and dev data...')
        logging.info('Training and dev data have been loaded.')
        train_df = pd.read_csv(config_dict["train_path"], index_col=0)
        with open(config_dict["dev_path"], 'rb') as f:
            dev_data = cPickle.load(f)
        train_model(train_df, dev_data)  # 训练


if __name__ == '__main__':
    main()
