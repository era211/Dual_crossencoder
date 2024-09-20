import os
import sys
import random
import logging
import argparse
import traceback
import pandas as pd
import pickle
import _pickle as cPickle
import json
from scorer import *  # calculate coref metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from transformers import RobertaTokenizer
#  Dynamically adding module search paths
from classes import *  # make sure classes in "/src/shared/" can be imported.
from fine_cf import *

parser = argparse.ArgumentParser(description='Training a cross-encoder')
parser.add_argument('--config_path',
                    type=str,
                    help=' The path configuration json file')

parser.add_argument('--out_dir',
                    type=str,
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
                    default=0,
                    help=' A single GPU number')
parser.add_argument('--load_data', default=True, type=bool, help='load data')
parser.add_argument('--load_test_data', default=True, type=bool, help='load test data')
parser.add_argument('--save_data', default=False, type=bool, help='save data')
# load all config parameters for training
args = parser.parse_args()
assert args.mode in ['train', 'eval'], "mode must be train or eval!"
# In the `train' mode, the best model and the evaluation results of the dev corpus are saved.
# In the `eval' mode, evaluation results of the test corpus are saved.


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
    print('Training with CUDA')
else:
    args.use_cuda = False

# Set global variables
best_score = None
patience = 0  # for early stopping
comparison_set = set()
tokenizer = RobertaTokenizer.from_pretrained("/root/autodl-tmp/PLM/roberta-base")  # tokenizer

# use train/dev data in 'train' mode, and use test data in 'eval' mode
print('Loading fixed dev set')
ecb_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/ecb/dev/dev_pairs')
fcc_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/fcc/dev/dev_pairs')
gvc_dev_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/gvc/dev/dev_pairs')

print('Loading fixed test set')
ecb_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/ecb/test/test_pairs')
fcc_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/fcc/test/test_pairs')
gvc_test_set = pd.read_pickle('/root/autodl-tmp/Rationale4CDECR-main/retrieved_data/main/gvc/test/test_pairs')

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
                the offset of the mention sentence in the window)
    '''
    lookback = max(0, sentence_id - window)
    lookforward = min(sentence_id + window, max(sentences.keys())) + 1
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
        sents_1, sent_id_1 = get_sents(doc_dict[mention_1.doc_id].sentences,
                                       mention_1.sent_id, window)
        sents_2, sent_id_2 = get_sents(doc_dict[mention_2.doc_id].sentences,
                                       mention_2.sent_id, window)
        tokens, token_map, offset_1, offset_2 = tokenize_and_map_pair(
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer)
        start_piece_1 = token_map[offset_1 + mention_1.start_offset][0]
        if offset_1 + mention_1.end_offset in token_map:
            end_piece_1 = token_map[offset_1 + mention_1.end_offset][-1]
        else:
            end_piece_1 = token_map[offset_1 + mention_1.start_offset][-1]
        start_piece_2 = token_map[offset_2 + mention_2.start_offset][0]
        if offset_2 + mention_2.end_offset in token_map:
            end_piece_2 = token_map[offset_2 + mention_2.end_offset][-1]
        else:
            end_piece_2 = token_map[offset_2 + mention_2.start_offset][-1]
        label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]
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
    assert eval_set in ['dev', 'test'], "please check the eval_set!"
    processed_dataset = []
    doc_dict = {
        key: document
        for topic in data_set.topics.values()
        for key, document in topic.docs.items()
    }
    train_set_name = config_dict["training_dataset"]
    test_set_name = config_dict["test_dataset"]
    if eval_set == 'dev':
        # even in ood test, dev and train set are from the same corpus.
        pairs = nn_generated_fixed_eval_pairs[train_set_name][eval_set]
    elif eval_set == 'test':
        pairs = nn_generated_fixed_eval_pairs[test_set_name][eval_set]
    pairs = list(pairs)
    for mention_1, mention_2 in pairs:
        record = structure_pair(mention_1, mention_2, doc_dict)
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
    return tensor_dataset, pairs, doc_dict


def structure_data_for_train(df):
    max_seq_length = 512
    all_data_index = df.index
    all_labels, all_sentences = [], []
    all_start_piece_1, all_end_piece_1 = [], []
    all_start_piece_2, all_end_piece_2 = [], []
    for ID in all_data_index:
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
        new_tokens = tokenizer.convert_ids_to_tokens(embeddings)
        total_tokens_num = df['total_tokens_num'][ID]
        token_map = dict(list(map(lambda x: (x, []), np.arange(total_tokens_num))))
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
        start_piece_1 = token_map[trigger_1_abs_start][0]
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
    return data_set_in_tensor



def structure_dataset_for_eval1(data_set,
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

def structure_data_for_train1(df):  # 构建训练数据对
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