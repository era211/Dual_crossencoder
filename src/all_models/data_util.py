import torch
import numpy as np
from tqdm import tqdm, trange
from absa_parser import headparser
from transformers import RobertaTokenizer, AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
tokenizer = RobertaTokenizer.from_pretrained("/home/yaolong/PT_MODELS/PT_MODELS/roberta-base")  # tokenizer

from torch.utils.data import Dataset, DataLoader
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
    print('构造训练数据...')
    max_seq_length = 512
    all_data_index = df.index
    all_labels, all_sentences = [], []
    all_start_piece_1, all_end_piece_1 = [], []
    all_start_piece_2, all_end_piece_2 = [], []
    all_syn_adj_ment1 = []
    all_syn_adj_ment2 = []
    ment_sentences = []
    for ID in tqdm(all_data_index[:100], desc='structure_train'):
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
        syn_adj_ment2 = syn_sent1(str(ment_sent_1))

        all_sentences.append(embeddings)  # 两个句子的编码
        all_start_piece_1.append([start_piece_1])
        all_end_piece_1.append([end_piece_1])
        all_start_piece_2.append([start_piece_2])
        all_end_piece_2.append([end_piece_2])
        all_labels.append(label)
        all_syn_adj_ment1.append(syn_adj_ment1)
        all_syn_adj_ment2.append(syn_adj_ment2)
        ment_sentences.append([str(ment_sent_1) + '<SEP> ' + str(ment_sent_2)])


    data_set_in_tensor = TensorDataset(torch.tensor(all_sentences), torch.tensor(all_start_piece_1), \
                                       torch.tensor(all_end_piece_1), torch.tensor(all_start_piece_2), \
                                       torch.tensor(all_end_piece_2), torch.tensor(all_labels))

    # 创建 PairedDataset 实例
    paired_dataset = PairedDataset(data_set_in_tensor, all_syn_adj_ment1, all_syn_adj_ment2)
    print("训练数据读取完成")
    return paired_dataset


def pad_to_size(array, target_size):
    padded_array = np.pad(array, (0, target_size - array.shape[0]), 'constant')
    return padded_array

def collate_fn(batch):
    sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels, all_syn_adj_1, all_syn_adj_2  = zip(*batch)

    # 对 all_syn_adj 进行 padding
    target_size = 100
    padded_all_syn_adj_1 = [torch.tensor(pad_to_size(arr, target_size)) for arr in all_syn_adj_1]
    padded_all_syn_adj_2 = [torch.tensor(pad_to_size(arr, target_size)) for arr in all_syn_adj_2]


    # 将每个部分堆叠为张量
    return (
        torch.stack(sentences),
        torch.stack(start_pieces_1),
        torch.stack(end_pieces_1),
        torch.stack(start_pieces_2),
        torch.stack(end_pieces_2),
        torch.stack(labels),
        torch.stack(padded_all_syn_adj_1),
        torch.stack(padded_all_syn_adj_2)
    )

