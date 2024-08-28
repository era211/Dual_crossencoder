import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from DualGCN_crossencoder_trainer import *

logging.info('Loading training and dev data...')
logging.info('Training and dev data have been loaded.')
train_df = pd.read_csv(config_dict["train_path"], index_col=0)
with open(config_dict["dev_path"], 'rb') as f:
    dev_data = cPickle.load(f)


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


def syn_sent1(sentence):
    from absa_parser import headparser
    headp, syntree = headparser.parse_heads(sentence)  # 返回句法树和依赖关系弧矩阵
    ori_adj = softmax1(headp[0])
    ori_adj = np.delete(ori_adj, 0, axis=0)
    ori_adj = np.delete(ori_adj, 0, axis=1)  # 删除了矩阵 ori_adj 的第一行和第一列，依存关系弧矩阵的第一个元素代表根节点或特定的虚拟节点，而这些节点在计算中可能是不必要的
    ori_adj -= np.diag(
        np.diag(ori_adj))  # 从 ori_adj 矩阵中减去其对角线上的元素。这样做的目的是将对角线上的值（通常表示一个词与自身的依赖关系）归零，确保在接下来的操作中仅考虑词与其他词的依赖关系。
    # if not opt.direct:  # 检查是否使用有向图或者无向图
    ori_adj = ori_adj + ori_adj.T  # 如果 opt.direct 为 False，表示使用无向图。在这种情况下，这行代码将矩阵 ori_adj 与它的转置矩阵 ori_adj.T 相加，从而将其转换为对称矩阵。这意味着如果 ori_adj[i][j] 有值，那么 ori_adj[j][i] 也会有相同的值，表示无向的依赖关系
    ori_adj = ori_adj + np.eye(ori_adj.shape[0])  # 向矩阵 ori_adj 的对角线位置添加 1，可以视为在图结构中，每个节点自身都有一个环
    # assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, ori_adj.shape)
    dj = np.pad(ori_adj, (0, 512 - ori_adj.shape[0]), 'constant')
    return dj

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


def structure_data_for_train(train_df):
    max_seq_length = 512
    all_data_index = train_df.index
    all_labels, all_sentences = [], []
    all_start_piece_1, all_end_piece_1 = [], []
    all_start_piece_2, all_end_piece_2 = [], []
    all_syn_adj = []
    for ID in tqdm(all_data_index[:300], desc='structure_train'):
        # get 'label'
        label = [float(train_df['label'][ID])]
        # get 'sentence'
        sentences_text_1 = train_df['text_1'][ID]
        sentences_text_2 = train_df['text_2'][ID]
        text_1_length = len(sentences_text_1.split(' '))
        embeddings = tokenizer(sentences_text_1,
                               sentences_text_2,
                               max_length=max_seq_length,
                               truncation=True,
                               padding="max_length")["input_ids"]
        # get start/end_piece_1/2
        counter = 0
        new_tokens = tokenizer.convert_ids_to_tokens(embeddings)  # 将编码数字转换为对应的token
        total_tokens_num = train_df['total_tokens_num'][ID]
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
        trigger_1_abs_start = train_df['trigger_1_abs_start'][ID]
        trigger_1_abs_end = train_df['trigger_1_abs_end'][ID]
        trigger_2_abs_start = train_df['trigger_2_abs_start'][ID]
        trigger_2_abs_end = train_df['trigger_2_abs_end'][ID]
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
        syn_adj = syn_sent1(str(ment_sent_1) +' '+ str(ment_sent_2))
        all_sentences.append(embeddings)  # 两个句子的编码
        all_start_piece_1.append([start_piece_1])
        all_end_piece_1.append([end_piece_1])
        all_start_piece_2.append([start_piece_2])
        all_end_piece_2.append([end_piece_2])
        all_labels.append(label)
        all_syn_adj.append(syn_adj)
    data_set_in_tensor = TensorDataset(torch.tensor(all_sentences), torch.tensor(all_start_piece_1), \
                                       torch.tensor(all_end_piece_1), torch.tensor(all_start_piece_2), \
                                       torch.tensor(all_end_piece_2), torch.tensor(all_labels), torch.tensor(all_syn_adj))


    # 保存 TensorDataset 为 .pkl 文件
    print("保存为文件...")
    with open('/home/yaolong/Rationale4CDECR-main/data_preparation/data_set_in_tensor_512.pkl', 'wb') as f:
        pickle.dump(data_set_in_tensor, f)
    print("文件保存成功...")

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

        tokens, token_map, offset_1, offset_2,  mention_sent_1, mention_sent_2 = tokenize_and_map_pair(
            # tokens是mention上下文窗口中所有原始句子编码后的token数字序列，token_map是原始句子中每个单词索引和编码的各个token索引之间的映射，其他两个是mention所在句子在token序列中的对应位置
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer)

        syn_adj = syn_sent1(mention_sent_1+ ''+ mention_sent_2)
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
            "end_piece_2": [end_piece_2],
            "syn_adj": syn_adj
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
    }  # dataset用来构建原始数据集中所有主题的文档字典
    train_set_name = config_dict["training_dataset"]
    test_set_name = config_dict["test_dataset"]
    if eval_set == 'dev':
        # even in ood test, dev and train set are from the same corpus.
        pairs = nn_generated_fixed_eval_pairs[train_set_name][eval_set]
    elif eval_set == 'test':
        pairs = nn_generated_fixed_eval_pairs[test_set_name][
            eval_set]  # 字典类型 /retrieved_data/main/ecb/test/test_pairs'  这个数据集中保存的都是mention对
    pairs = list(pairs)  # 将字典类型转换为列表类型
    for mention_1, mention_2 in tqdm(pairs):  # 从提及对数据列表中分别读取每一对mention
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
    all_syn_adj = torch.tensor(
        [record["syn_adj"] for record in processed_dataset])
    print(labels.sum() / float(labels.shape[0]))
    tensor_dataset = TensorDataset(sentences, start_pieces_1, end_pieces_1, \
                                   start_pieces_2, end_pieces_2, labels, all_syn_adj)
    return tensor_dataset, pairs, doc_dict

if __name__ == '__main__':
    structure_data_for_train(train_df)
