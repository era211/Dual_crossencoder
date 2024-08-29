'''
1.在这一个版本中，构造语法图的时候，将对每个提及进行分别构造，然后分别进行卷积操作
2.将图卷积的操作长度进行限制
'''
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import logging
import itertools
from tqdm import tqdm, trange
# --- data process
import pickle
import pickle as cPickle
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
from transformers.optimization import get_linear_schedule_with_warmup

#  Dynamically adding module search paths
src_dir = "/home/yaolong/Rationale4CDECR-main/src"

for pack in os.listdir(src_dir):
    full_path = os.path.join(src_dir, pack)
    if os.path.isdir(full_path):  # 仅在 pack 是目录时才添加到 sys.path
        sys.path.append(full_path)
sys.path.append("/home/yaolong/Rationale4CDECR-main/src/shared/")
sys.path.append("/home/yaolong/Rationale4CDECR-main/src/LAL_Parser/src_joint/")
from data_util import *
from classes import *  # make sure classes in "/src/shared/" can be imported.
from bcubed_scorer import *
from coarse import *
from fine_v4 import *


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
tokenizer = RobertaTokenizer.from_pretrained("/home/yaolong/PT_MODELS/PT_MODELS/roberta-base")  # tokenizer
# 这个tokenizer在哪里使用了：构建训练句子对的时候，对句子对进行token化


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


def get_bert_optimizer(model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["roberta.embeddings", "roberta.encoder"]

    if args.diff_lr:
        logging.info("layered learning rate on")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": args.weight_decay,
                "lr": args.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": args.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": args.learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    else:
        logging.info("bert learning rate on")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_lr, eps=args.adam_epsilon)

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
            comparison_set = comparison_set | set(
                [frozenset([mention_id_1, mention_id_2])])
            mention_2 = mentions[mention_id_2]
            record = structure_pair_dual(mention_1, mention_2, doc_dict)
            records.append(record)
        sentences = torch.tensor([record["sentence"]
                                  for record in records])
        labels = torch.tensor([record["label"]
                               for record in records])
        start_pieces_1 = torch.tensor(
            [record["start_piece_1"] for record in records])
        end_pieces_1 = torch.tensor(
            [record["end_piece_1"] for record in records])
        start_pieces_2 = torch.tensor(
            [record["start_piece_2"] for record in records])
        end_pieces_2 = torch.tensor(
            [record["end_piece_2"] for record in records])
        embeddings_dev_ment1 = torch.tensor(
            [record["embeddings_dev_ment1"] for record in records])
        embeddings_dev_ment2 = torch.tensor(
            [record["embeddings_dev_ment2"] for record in records])
        all_syn_adj_1 = [record["syn_adj_1"] for record in records]
        all_syn_adj_2 = [record["syn_adj_2"] for record in records]
        tensor_dataset_dev = TensorDataset(sentences, start_pieces_1, end_pieces_1,
                                           start_pieces_2, end_pieces_2, embeddings_dev_ment1, embeddings_dev_ment2, labels)

        # 创建 PairedDataset 实例
        paired_dataset = PairedDataset(tensor_dataset_dev, all_syn_adj_1, all_syn_adj_2)
        dev_dataloader = DataLoader(paired_dataset,
                                    sampler=SequentialSampler(paired_dataset),
                                    batch_size=len(all_syn_adj_1))

        sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, all_embeddings_ment1, all_embeddings_ment2, labels, adj1, adj2 = dev_dataloader
        with torch.no_grad():
            out_dict = model(sentences, start_pieces_1, end_pieces_1,
                                start_pieces_2, end_pieces_2, adj1, adj2, all_embeddings_ment1, all_embeddings_ment2, labels)
            mean_prob = torch.mean(out_dict["probabilities"]).item()
            score += mean_prob
    print('判断是否将聚类合并完成：', (score / len(cluster_1)) >= 0.5)
    return (score / len(cluster_1)) >= 0.5


def transitive_closure_merge(edges, mentions, model, doc_dict, graph,
                             graph_render):  # 对共指提及进行聚类
    clusters = {}
    inv_clusters = {}
    mentions = {mention.mention_id: mention for mention in mentions}
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
def evaluate(model, dev_dataloader, dev_pairs, doc_dict, epoch_num):
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
        sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, all_embeddings_ment1, all_embeddings_ment2, labels, adj1, adj2 = batch
        if not config_dict["oracle"]:
            with torch.no_grad():
                out_dict = model(sentences, start_pieces_1, end_pieces_1,
                                        start_pieces_2, end_pieces_2, adj1, adj2, all_embeddings_ment1, all_embeddings_ment2, labels)
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
            comparison_set = comparison_set | set(
                [frozenset([pair_0.mention_id, pair_1.mention_id])])
            if probs[p_index][0] > 0.5:
                if pair_0.mention_id not in best_edges or (
                        probs[p_index][0] > best_edges[pair_0.mention_id][3]):
                    best_edges[pair_0.mention_id] = (pair_0.mention_id,
                                                     pair_1.mention_id,
                                                     labels[p_index][0],
                                                     probs[p_index][0])
                edges.add((pair_0.mention_id, pair_1.mention_id,
                           labels[p_index][0], probs[p_index][0]))
            saved_edges.append((pair_0, pair_1,
                                labels[p_index][0].detach().cpu().tolist(), probs[p_index][0]))
        offset += len(predictions)

    tqdm.write("验证集准确率：Pairwise Accuracy: {:.6f}".format(acc_sum /
                                                  float(len(dev_dataloader))))
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
    if args.mode == 'eval':  # During Test
        print('saving gold_map, model_map...')
        # save the golden_map which groups all mentions based on the annotation.
        with open(os.path.join(args.out_dir, "gold_map"), 'wb') as f:
            pickle.dump(gold_map, f)
            # save the model_map which groups all mentions based on the coreference pipeline.
        with open(os.path.join(args.out_dir, "model_map"), 'wb') as f:
            pickle.dump(model_map, f)
        print('Saved!')
    else:
        pass
    # deprecated
    # pn, pd = b_cubed(model_clusters, gold_map)
    # rn, rd = b_cubed(gold_clusters, model_map)
    # tqdm.write("Alternate = Recall: {:.6f} Precision: {:.6f}".format(pn/pd, rn/rd))
    p, r, f1 = bcubed(gold_sets, model_sets)
    tqdm.write("Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(p, r, f1))
    if best_score == None or f1 > best_score:
        tqdm.write("F1 Improved Saving Model")
        best_score = f1
        patience = 0
        if args.mode == 'train':  # During training
            # save the best model measured by b_cubded F1 in the current epoch
            print('保存模型参数')
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "AD_crossencoder_best_model"),
            )
            # save the edges linking coreferential events on the dev corpus
            with open(os.path.join(args.out_dir, "crossencoder_dev_edges"), "wb") as f:
                cPickle.dump(saved_edges, f)
        else:  # During Test
            # save the edges linking coreferential events on the test corpus
            with open(os.path.join(args.out_dir, "crossencoder_test_edges"), "wb") as f:
                cPickle.dump(saved_edges, f)
            # dot.render(os.path.join(args.out_dir, "clustering"))
    else:
        patience += 1
        if patience > config_dict["early_stop_patience"]:
            print("Early Stopping")
            sys.exit()


def train_model(df, dev_set):
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))
    print(f"Using device: {device}")
    # load bi-encoder model
    # event_encoder_path = config_dict['event_encoder_model']
    # with open(event_encoder_path, 'rb') as f:
    #     params = torch.load(f, map_location=device)  # 将读取到的数据加载到指定的设备上，默认为0卡
    #     event_encoder = EncoderCosineRanker(device)
    #     event_encoder.load_state_dict(params)
    #     event_encoder = event_encoder.to(device).eval()
    #     event_encoder.requires_grad = False
    # model = CoreferenceCrossEncoder(device).to(device)
    model = None
    model_dual = CoreferenceCrossEncoder_DualGCN(device, args=args).to(device)
    train_data_num = len(df)  # baseline: 49864

    train_event_pairs = structure_data_for_train(df)  # 构建训练数据集，TensorDate类，返回值包括所有训练数据的embedding，触发词的开始和结束位置以及label

    dev_event_pairs, dev_pairs, dev_docs = structure_dataset_for_eval(dev_set, eval_set='dev')
    print('开始训练...')
    if args.DualGCN:
        optimizer = get_bert_optimizer(model_dual)
    else:
        optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_data_num)
    train_dataloader = DataLoader(train_event_pairs,
                                  batch_size=config_dict["batch_size"],
                                  shuffle=True,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_event_pairs,
                                sampler=SequentialSampler(dev_event_pairs),
                                batch_size=config_dict["batch_size"])
    for epoch_idx in trange(int(config_dict["epochs"]),
                            desc="Epoch",
                            leave=True):
        if args.DualGCN:
            model_dual = model_dual.train()
        else:
            model = model.train()
        tr_loss = 0.0
        tr_p = 0.0
        tr_a = 0.0
        tr_r = 0.0
        tr_f = 0.0
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(device) for t in batch)

            sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, all_embeddings_ment1, all_embeddings_ment2, labels, adj1, adj2 = batch
            if args.DualGCN:
                out_dict = model_dual(sentences, start_pieces_1, end_pieces_1,
                                             start_pieces_2, end_pieces_2, adj1, adj2, all_embeddings_ment1, all_embeddings_ment2, labels)
            else:
                penal = None
                out_dict = model(sentences, start_pieces_1, end_pieces_1,
                                 start_pieces_2, end_pieces_2, labels)
            loss = out_dict["loss"]
            precision = out_dict["precision"]
            accuracy = out_dict["accuracy"]
            recall = out_dict["recall"]
            f1 = out_dict["f1_score"]
            loss.backward()
            tr_loss += loss.item()
            tr_p += precision.item()
            tr_a += accuracy.item()
            tr_r += recall
            tr_f += f1
            if ((step + 1) * config_dict["batch_size"]
            ) % config_dict["accumulated_batch_size"] == 0:
                # For main exp, we set batch_size as 10, accumulated_batch_size as 8.
                # This should be equivalent to the case of batch_size being 40 and accumulated_batch_size being 8
                batcher.set_description(
                    "Batch (average loss: {:.6f} accuracy: {:.6f} f1: {:.6f})"
                    .format(
                        tr_loss / float(step + 1),
                        tr_a / float(step + 1),
                        tr_f / float(step + 1),
                    ))
                torch.nn.utils.clip_grad_norm_(model_dual.parameters(),
                                               config_dict["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        evaluate(model_dual, dev_dataloader, dev_pairs, dev_docs, epoch_idx)


def main():
    if args.mode == 'train':
        print("模型训练！")
        logging.info('Loading training and dev data...')
        logging.info('Training and dev data have been loaded.')
        train_df = pd.read_csv(config_dict["train_path"], index_col=0)
        with open(config_dict["dev_path"], 'rb') as f:  # /all_sents/dev_data
            dev_data = cPickle.load(f)
        train_model(train_df, dev_data)
    elif args.mode == 'eval':
        print('eval')
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
        device = torch.device("cuda:5" if args.use_cuda else "cpu")
        event_encoder_path = config_dict['event_encoder_model']
        with open(event_encoder_path, 'rb') as f:
            params = torch.load(f, map_location=device)
            event_encoder = EncoderCosineRanker(device)
            event_encoder.load_state_dict(params)
            event_encoder = event_encoder.to(device).eval()
            event_encoder.requires_grad = False

        if config_dict['eval_model_path'] == False:
            logging.info('Loading default model for eval...')
            eval_model_path = os.path.join(args.out_dir, 'AD_crossencoder_best_model')
        else:
            logging.info('Loading the specified model for eval...')
            eval_model_path = config_dict['eval_model_path']
        with open(eval_model_path, 'rb') as f:
            params = torch.load(f)
            model = CoreferenceCrossEncoder(device)
            model.load_state_dict(params)
            model = model.to(device).eval()
            model.requires_grad = False
        test_event_pairs, test_pairs, test_docs = structure_dataset_for_eval(test_data,
                                                                             eval_set='test')  # test_event_pairs是编码后的上下文token数字序列以及标签，两个mention的起始和结束索引，test_pairs是mention对，test_docs是文档字典
        test_dataloader = DataLoader(test_event_pairs,
                                     sampler=SequentialSampler(test_event_pairs),  # 顺序采样
                                     batch_size=config_dict["batch_size"])
        evaluate(model, event_encoder, test_dataloader, test_pairs, test_docs, 0)


if __name__ == '__main__':
    main()
