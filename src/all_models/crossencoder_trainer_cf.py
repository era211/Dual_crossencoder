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
import _pickle as cPickle
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
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))
sys.path.append("/root/autodl-tmp/Rationale4CDECR-main/src/shared/")
from classes import *  # make sure classes in "/src/shared/" can be imported.
from bcubed_scorer import *
from coarse import *
from data_util_cf import *


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
            comparison_set = comparison_set | set(
                [frozenset([mention_id_1, mention_id_2])])
            mention_2 = mentions[mention_id_2]
            record = structure_pair(mention_1, mention_2, doc_dict)
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
        # 将这些数据组合成一个 batch（注意使用 zip 打包）
        batch = list(zip(sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels))
        c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = collate_fn(batch)
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
                             graph_render):
    clusters = {}
    inv_clusters = {}
    mentions = {mention.mention_id: mention for mention in mentions}
    for edge in tqdm(edges):
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
                graph.add_edge(edge[0], edge[1])
                color = 'black'
                if edge[2] != 1.0:
                    color = 'red'
                graph_render.edge(edge[0],
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
        c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
        if not config_dict["oracle"]:
            with torch.no_grad():
                out_dict = model(c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1,
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
        for p_index in range(len(predictions)):
            pair_0, pair_1 = dev_pairs[offset + p_index]
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

    tqdm.write("Pairwise Accuracy: {:.6f}".format(acc_sum /
                                                  float(len(dev_dataloader))))
    # writer.add_scalar('dev/pairwise_acc',acc_sum/float(len(dev_dataloader)),epoch_num)
    eval_edges(edges, mentions, model, doc_dict, saved_edges)
    assert len(saved_edges) >= len(edges)
    return saved_edges


# eval the coref-metric based on edges among clusters
def eval_edges(edges, mentions, model, doc_dict, saved_edges):
    print(len(mentions))
    global best_score, patience
    dot = Graph(comment='Cross Doc Co-ref')
    G = nx.Graph()
    edges = sorted(edges, key=lambda x: -1 * x[3])
    for mention in mentions:
        G.add_node(mention.mention_id)
        dot.node(mention.mention_id,
                 label=str((str(mention), doc_dict[mention.doc_id].sentences[
                     mention.sent_id].get_raw_sentence())))
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))
    # edges = [edge for edge in edges if edge not in bridges]
    clusters, inv_clusters = transitive_closure_merge(edges, mentions, model,
                                                      doc_dict, G, dot)

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
            model_map[mention.mention_id] = mention.mention_id
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
    # load bi-encoder model
    event_encoder_path = config_dict['event_encoder_model']
    # with open(event_encoder_path, 'rb') as f:
    #     params = torch.load(f)
    #     event_encoder = EncoderCosineRanker(device)
    #     event_encoder.load_state_dict(params)
    #     event_encoder = event_encoder.to(device).eval()
    #     event_encoder.requires_grad = False
    event_encoder = None
    model = CoreferenceCrossEncoder(device).to(device)
    train_data_num = len(df)
    train_event_pairs = structure_data_for_train1(df)
    dev_event_pairs, dev_pairs, dev_docs = structure_dataset_for_eval1(dev_set, eval_set='dev')
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
        model = model.train()
        tr_loss = 0.0
        tr_p = 0.0
        tr_a = 0.0
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(device) for t in batch)
            c_sentences, e_sentences, sentences, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
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
        evaluate(model, event_encoder, dev_dataloader, dev_pairs, dev_docs,
                 epoch_idx)


def main():
    if args.mode == 'train':
        logging.info('Loading training and dev data...')
        logging.info('Training and dev data have been loaded.')
        train_df = pd.read_csv(config_dict["train_path"], index_col=0)
        with open(config_dict["dev_path"], 'rb') as f:
            dev_data = cPickle.load(f)
        train_model(train_df, dev_data)
    elif args.mode == 'eval':
        with open(config_dict["test_path"], 'rb') as f:
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
        device = torch.device("cuda:0" if args.use_cuda else "cpu")
        event_encoder_path = config_dict['event_encoder_model']
        with open(event_encoder_path, 'rb') as f:
            params = torch.load(f)
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
        test_event_pairs, test_pairs, test_docs = structure_dataset_for_eval(test_data, eval_set='test')
        test_dataloader = DataLoader(test_event_pairs,
                                     sampler=SequentialSampler(test_event_pairs),
                                     batch_size=config_dict["batch_size"])
        evaluate(model, event_encoder, test_dataloader, test_pairs, test_docs, 0)


if __name__ == '__main__':
    main()
