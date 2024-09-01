import json
import argparse
import traceback
import pickle
import pickle as cPickle
import pandas as pd
from tqdm import tqdm, trange
from absa_parser import headparser
from transformers import BertTokenizer
from transformers import RobertaTokenizer, AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
tokenizer = RobertaTokenizer.from_pretrained("/home/yaolong/PT_MODELS/PT_MODELS/roberta-base")  # tokenizer
tokenizr_bert = BertTokenizer.from_pretrained('/home/yaolong/PT_MODELS/PT_MODELS/bert-base-uncased')  # max_length=100, dem=768
from fine_v4 import *
from torch.utils.data import Dataset, DataLoader

'''添加参数'''
parser = argparse.ArgumentParser(description='Training a cross-encoder')
parser.add_argument('--config_path',
                    type=str,
                    default='/home/yaolong/Rationale4CDECR-main/configs/main/ecb/baseline.json',
                    help=' The path configuration json file')

parser.add_argument('--out_dir',
                    type=str,
                    default='/home/yaolong/Rationale4CDECR-main/outputs/main/ecb/baseline/dual/dual_save_data/best_model',
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

# DualGCN
parser.add_argument('--DualGCN',
                    type=bool,
                    default=True,
                    help=' use DualGCN')
parser.add_argument('--num_layers',
                    type=int,
                    default=2,
                    help='Num of GCN layers.')
parser.add_argument('--bert_dropout',
                    type=float,
                    default=0.3,
                    help='BERT dropout rate.')
parser.add_argument('--gcn_dropout',
                    type=float,
                    default=0.1,
                    help='GCN layer dropout rate.')
parser.add_argument('--bert_dim',
                    type=int,
                    default=1024,
                    help='RoBERTa-large embedding size.')
parser.add_argument('--attention_heads',
                    type=int,
                    default=1,
                    help='number of multi-attention heads')
parser.add_argument('--losstype',
                    type=str,
                    default='differentiatedloss',
                    help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
parser.add_argument('--load_data', default=True, type=bool, help='load data')
parser.add_argument('--save_data', default=False, type=bool, help='load data')
parser.add_argument('--alpha', default=0.25, type=float)
parser.add_argument('--use_cuda', default=True, type=bool, help='use gpu')
parser.add_argument('--beta', default=0.25, type=float)
parser.add_argument('--penal_alpha', default=0.25, type=float)
parser.add_argument('--penal_beta', default=0.25, type=float)
parser.add_argument('--diff_lr', default=False, action='store_true')
parser.add_argument('--bert_lr', default=1e-4, type=float) # 1e-4
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--adj_pad_size', default=100, type=float, help="adj pad size")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--weight_decay", default=1e-8, type=float, help="Weight deay if we apply some.")

# load all config parameters for training
args = parser.parse_args()
# use train/dev data in 'train' mode, and use test data in 'eval' mode
assert args.mode in ['train', 'eval'], "mode must be train or eval!"
# In the `train' mode, the best model and the evaluation results of the dev corpus are saved.
# In the `eval' mode, evaluation results of the test corpus are saved.

# 读取配置文件
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

'''读取数据'''
print('Loading fixed dev set')
# 这些数据对的产生过程
ecb_dev_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/ecb/dev/dev_pairs')
fcc_dev_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/fcc/dev/dev_pairs')
gvc_dev_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/gvc/dev/dev_pairs')

print('Loading fixed test set')
ecb_test_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/ecb/test/test_pairs')
fcc_test_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/fcc/test/test_pairs')
gvc_test_set = pd.read_pickle('/home/yaolong/Rationale4CDECR-main/retrieved_data/main/gvc/test/test_pairs')
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


'''自定义数据类'''
class PairedDataset(Dataset):
    def __init__(self, data_set_in_tensor, all_syn_adj_1, all_syn_adj_2):
        self.data_set_in_tensor = data_set_in_tensor
        self.all_syn_adj_1 = all_syn_adj_1
        self.all_syn_adj_2 = all_syn_adj_2


    def __len__(self):
        # 假设两个张量的长度相同
        return len(self.all_syn_adj_1)

    def __getitem__(self, idx):
        # 返回两个张量中对应的数据对
        data = self.data_set_in_tensor[idx]
        syn_adj_1 = self.all_syn_adj_1[idx]
        syn_adj_2 = self.all_syn_adj_2[idx]
        return *data, syn_adj_1, syn_adj_2

def softmax1(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


'''构造train数据'''
def get_target_sentence(sentence, trigger_id):
    import nltk
    # nltk.download('punkt')
    text = sentence
    sentences = nltk.sent_tokenize(text)
    # print(sentences)
    # 给定索引位置
    word_index = trigger_id

    # 迭代句子以找到索引位置
    current_index = 0
    target_sentence = None
    count = 0

    for sentence in sentences:
        if count == 68:
            print('1')
            print(sentence)
        if sentence is None:
            continue  # 如果句子为 None，跳过

        sentence_length = len(sentence)
        if current_index <= word_index < current_index + sentence_length:
            target_sentence = sentence
            break
        current_index += sentence_length + 1  # 加1是因为句子之间有空格分隔
        count += 1

    # print(target_sentence)
    return target_sentence


def syn_sent1(sentence):
    headp, syntree = headparser.parse_heads(sentence)  # 返回依赖关系弧矩阵和句法树
    ori_adj = softmax1(headp[0])
    ori_adj = np.delete(ori_adj, 0, axis=0)
    ori_adj = np.delete(ori_adj, 0, axis=1)  # 删除了矩阵 ori_adj 的第一行和第一列，依存关系弧矩阵的第一个元素代表根节点或特定的虚拟节点，而这些节点在计算中可能是不必要的
    ori_adj = ori_adj - np.diag(
        np.diag(ori_adj))  # 从 ori_adj 矩阵中减去其对角线上的元素。这样做的目的是将对角线上的值（通常表示一个词与自身的依赖关系）归零，确保在接下来的操作中仅考虑词与其他词的依赖关系。
    # if not opt.direct:  # 检查是否使用有向图或者无向图
    ori_adj = ori_adj + ori_adj.T  # 如果 opt.direct 为 False，表示使用无向图。在这种情况下，这行代码将矩阵 ori_adj 与它的转置矩阵 ori_adj.T 相加，从而将其转换为对称矩阵。这意味着如果 ori_adj[i][j] 有值，那么 ori_adj[j][i] 也会有相同的值，表示无向的依赖关系
    ori_adj = ori_adj + np.eye(ori_adj.shape[0])  # 向矩阵 ori_adj 的对角线位置添加 1，可以视为在图结构中，每个节点自身都有一个环
    # assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, ori_adj.shape)
    return ori_adj


def structure_data_for_train(df):  # 构建训练数据对
    if not args.load_data:
        print('构造训练数据...')
        max_seq_length = 512
        max_seq_length_ment = 100
        all_data_index = df.index
        all_labels, all_sentences = [], []
        all_start_piece_1, all_end_piece_1 = [], []
        all_start_piece_2, all_end_piece_2 = [], []
        all_syn_adj_ment1 = []
        all_syn_adj_ment2 = []
        all_embeddings_ment1 = []
        all_embeddings_ment2 = []
        ment_sentences = []
        for ID in tqdm(all_data_index[:50], desc='structure_train'):
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
            ment_sent_1 = get_target_sentence(sentences_text_1, trigger_1_abs_start)
            ment_sent_2 = get_target_sentence(sentences_text_2, trigger_2_abs_start)

            syn_adj_ment1 = syn_sent1(str(ment_sent_1))
            syn_adj_ment2 = syn_sent1(str(ment_sent_2))

            all_sentences.append(embeddings)  # 两个句子的编码
            all_start_piece_1.append([start_piece_1])
            all_end_piece_1.append([end_piece_1])
            all_start_piece_2.append([start_piece_2])
            all_end_piece_2.append([end_piece_2])
            all_labels.append(label)
            all_syn_adj_ment1.append(syn_adj_ment1)
            all_syn_adj_ment2.append(syn_adj_ment2)
            # ment_sentences.append([str(ment_sent_1) + '<SEP> ' + str(ment_sent_2)])

            embeddings_ment1 = tokenizer(str(ment_sent_1),
                                   max_length=max_seq_length_ment,
                                   truncation=True,
                                   padding="max_length")["input_ids"]
            embeddings_ment2 = tokenizer(str(ment_sent_2),
                                   max_length=max_seq_length_ment,
                                   truncation=True,
                                   padding="max_length")["input_ids"]

            all_embeddings_ment1.append(embeddings_ment1)
            all_embeddings_ment2.append(embeddings_ment2)


        data_set_in_tensor = TensorDataset(torch.tensor(all_sentences), torch.tensor(all_start_piece_1),
                                           torch.tensor(all_end_piece_1), torch.tensor(all_start_piece_2),
                                           torch.tensor(all_end_piece_2), torch.tensor(all_embeddings_ment1),
                                           torch.tensor(all_embeddings_ment2), torch.tensor(all_labels))
        if args.save_data:
            # 将数据打包进一个字典
            data_dict = {
                'all_syn_adj_ment1': all_syn_adj_ment1,
                'all_syn_adj_ment2': all_syn_adj_ment2,
                'data_set_in_tensor': data_set_in_tensor
                }

            # 保存字典到 .pkl 文件
            file_path = '/home/yaolong/Rationale4CDECR-main/data_preparation/1/train_data.pkl'  # 文件路径
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print("数据已成功保存到文件。")

    else:
        print('加载数据...')
        # 指定文件路径
        file_path = '/home/yaolong/Rationale4CDECR-main/data_preparation/train_data.pkl'

        # 打开文件并加载数据
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 现在可以从字典中获取各个数据
        all_syn_adj_ment1 = data_dict['all_syn_adj_ment1']
        all_syn_adj_ment2 = data_dict['all_syn_adj_ment2']
        data_set_in_tensor = data_dict['data_set_in_tensor']

        print("数据已成功从文件中读取...")

    # 创建 PairedDataset 实例
    paired_dataset = PairedDataset(data_set_in_tensor, all_syn_adj_ment1, all_syn_adj_ment2)
    print("训练数据读取完成")
    return paired_dataset


'''构造dev数据'''
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


def structure_pair_dual(mention_1, mention_2, doc_dict, window=config_dict["window_size"]):
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

        tokens, token_map, offset_1, offset_2, mention_sent_1, mention_sent_2,  embeddings_dev_ment1, embeddings_dev_ment2 = tokenize_and_map_pair(
            # tokens是mention上下文窗口中所有原始句子编码后的token数字序列，token_map是原始句子中每个单词索引和编码的各个token索引之间的映射，其他两个是mention所在句子在token序列中的对应位置
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer)

        syn_adj_1 = syn_sent1(mention_sent_1)
        syn_adj_2 = syn_sent1(mention_sent_2)

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
            "id": [id],
            "sentence": tokens,
            # the embedding of pairwise mention data, i.e., tokenizer(sent1_with_discourse, sent2_with_discourse)
            "label": label,  # coref (1.0) or non-coref (0.0)
            "start_piece_1": [start_piece_1],  # the start and end offset of trigger_1 pieces in "sentence"
            "end_piece_1": [end_piece_1],
            "start_piece_2": [start_piece_2],  # # the start and end offset of trigger_2 pieces in "sentence"
            "end_piece_2": [end_piece_2],
            "syn_adj_1": syn_adj_1,
            "syn_adj_2": syn_adj_2,
            "embeddings_dev_ment1": embeddings_dev_ment1,
            "embeddings_dev_ment2": embeddings_dev_ment2,
            "ment_sentences": mention_sent_1 + '<SEP>' + mention_sent_2
        }
    except:
        if window > 0:
            return structure_pair_dual(mention_1, mention_2, doc_dict, window - 1)
        else:
            traceback.print_exc()
            sys.exit()
    return record


def structure_dataset_for_eval(data_set, eval_set='dev'):
    assert eval_set in ['dev', 'test'], "please check the eval_set!"
    processed_dataset = []
    doc_dict = {
        key: document
        for topic in data_set.topics.values()
        for key, document in topic.docs.items()
    }  # dataset用来构建原始数据集中所有主题的文档字典
    train_set_name = config_dict["training_dataset"]
    test_set_name = config_dict["test_dataset"]
    if eval_set == 'dev':
        # even in ood test, dev and train set are from the same corpus.
        pairs = nn_generated_fixed_eval_pairs[train_set_name][eval_set]
    elif eval_set == 'test':
        pairs = nn_generated_fixed_eval_pairs[test_set_name][eval_set]  # 字典类型 /retrieved_data/main/ecb/test/test_pairs'  这个数据集中保存的都是mention对
    pairs = list(pairs)  # 将字典类型转换为列表类型

    if not args.load_data:
        id = 0
        for mention_1, mention_2 in tqdm(pairs[:50]):  # 从提及对数据列表中分别读取每一对mention
            record = structure_pair_dual(mention_1, mention_2,
                                         doc_dict)  # 得到两个mention的上下文句子的token序列，标签值，mention分别在它们对应句子的token序列中的起始和结束位置
            processed_dataset.append(record)
            id = id + 1
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
        all_syn_adj_1 = [record["syn_adj_1"] for record in processed_dataset]
        all_syn_adj_2 = [record["syn_adj_2"] for record in processed_dataset]
        embeddings_dev_ment1 = torch.tensor(
            [record["embeddings_dev_ment1"] for record in processed_dataset])
        embeddings_dev_ment2 = torch.tensor(
            [record["embeddings_dev_ment2"] for record in processed_dataset])
        print(labels.sum() / float(labels.shape[0]))
        tensor_dataset_dev = TensorDataset(sentences, start_pieces_1, end_pieces_1,
                                       start_pieces_2, end_pieces_2, embeddings_dev_ment1, embeddings_dev_ment2, labels)

        if args.save_data:
            # 将数据打包进一个字典
            data_dict = {
                'all_syn_adj_1': all_syn_adj_1,
                'all_syn_adj_2': all_syn_adj_2,
                'tensor_dataset_dev': tensor_dataset_dev
                }

            # 保存字典到 .pkl 文件
            file_path = '/home/yaolong/Rationale4CDECR-main/data_preparation/test_data.pkl'  # 文件路径
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print("数据已成功保存到文件。")

    else:
        print('读取dev数据..')
        # 指定文件路径
        file_path = '/home/yaolong/Rationale4CDECR-main/data_preparation/dev_data.pkl'

        # 打开文件并加载数据
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 现在可以从字典中获取各个数据
        all_syn_adj_1 = data_dict['all_syn_adj_1']
        all_syn_adj_2 = data_dict['all_syn_adj_2']
        tensor_dataset_dev = data_dict['tensor_dataset_dev']

        print("数据已成功从文件中读取...")

    # 创建 PairedDataset 实例
    paired_dataset = PairedDataset(tensor_dataset_dev, all_syn_adj_1, all_syn_adj_2)
    print('验证集数据构造完成...')
    return paired_dataset, pairs, doc_dict

def pad_to_size(array, target_size):
    # padded_array = np.pad(array, (0, target_size - array.shape[0]), 'constant')
    current_size = array.shape[0]  # 获取当前行列数（因为是方阵，行列数相等）

    if target_size > current_size:
        # 对行和列进行填充，使其变为目标大小的方阵
        padded_array = np.pad(array, ((0, target_size - current_size), (0, target_size - current_size)), 'constant')
    elif target_size < current_size:
        # 对行和列进行截断
        padded_array = array[:target_size, :target_size]
    else:
        # 如果target_size等于current_size，保持原样
        padded_array = array

    return padded_array

def collate_fn(batch):
    sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, all_embeddings_ment1, all_embeddings_ment2, labels, all_syn_adj_1, all_syn_adj_2  = zip(*batch)

    # 对 all_syn_adj 进行 padding
    target_size = args.adj_pad_size
    padded_all_syn_adj_1 = [torch.tensor(pad_to_size(arr, target_size)) for arr in all_syn_adj_1]
    padded_all_syn_adj_2 = [torch.tensor(pad_to_size(arr, target_size)) for arr in all_syn_adj_2]


    # 将每个部分堆叠为张量
    return (
        torch.stack(sentences),
        torch.stack(start_pieces_1),
        torch.stack(end_pieces_1),
        torch.stack(start_pieces_2),
        torch.stack(end_pieces_2),
        torch.stack(all_embeddings_ment1),  # 100*1024
        torch.stack(all_embeddings_ment2),
        torch.stack(labels),
        torch.stack(padded_all_syn_adj_1),
        torch.stack(padded_all_syn_adj_2)
    )

