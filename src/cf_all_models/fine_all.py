import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import RobertaModel
from coarse import EncoderCosineRanker

def get_raw_strings(sentences, mention_sentence, mapping=None):
    raw_strings = []
    offset = 0
    if mapping:
        offset = max(mapping.keys()) + 1
    else:
        mapping = {}
    for i, sentence in enumerate(sentences):
        blacklist = []
        raw_strings.append(' '.join([
            tok.get_token().replace(" ", "") if
            (int(tok.token_id) not in blacklist) else "[MASK]"
            for tok in sentence.get_tokens()
        ]))   # 得到句子的原始表示
        if i == mention_sentence:  # 当前的句子就是mention所在的句子
            mention_offset = offset
        for _ in sentence.get_tokens_strings():  # 每个句子中的每个token对应一个mapping列表
            mapping[offset] = []
            offset += 1
    return raw_strings, mention_offset, mapping


def get_raw_strings1(sentences, mention_sentence, mapping=None):
    raw_strings = []
    offset = 0
    mention_sent = None
    if mapping:
        offset = max(mapping.keys()) + 1
    else:
        mapping = {}
    for i, sentence in enumerate(sentences):
        blacklist = []
        raw_strings.append(' '.join([
            tok.get_token().replace(" ", "") if
            (int(tok.token_id) not in blacklist) else "[MASK]"
            for tok in sentence.get_tokens()
        ]))   # 得到句子的原始表示
        if i == mention_sentence:  # 当前的句子就是mention所在的句子
            mention_offset = offset
            mention_sent = raw_strings[mention_sentence]
        for _ in sentence.get_tokens_strings():  # 每个句子中的每个token对应一个mapping列表
            mapping[offset] = []
            offset += 1
    return raw_strings, mention_offset, mapping, mention_sent


def tokenize_and_map_pair(sentences_1, sentences_2, mention_sentence_1,
                          mention_sentence_2, tokenizer):
    max_seq_length = 512
    raw_strings_1, mention_offset_1, mapping, mention_sent_1 = get_raw_strings1(
        sentences_1, mention_sentence_1)  # raw_strings_1是mention对应句子上下文窗口中的所有句子的原始内容，
    raw_strings_2, mention_offset_2, mapping, mention_sent_2 = get_raw_strings1(
        sentences_2, mention_sentence_2, mapping)
    embeddings = tokenizer(' '.join(raw_strings_1),
                           ' '.join(raw_strings_2),
                           max_length=max_seq_length,
                           truncation=True,
                           padding="max_length")["input_ids"]  # 将mention对的所有上下文窗口中的句子进行编码，得到token编码序列
    counter = 0
    new_tokens = tokenizer.convert_ids_to_tokens(embeddings)  # 将编码后的token编码序列转换成token
    for i, token in enumerate(new_tokens):  # 从new_tokens列表中一个一个取出token
        if (new_tokens[i]=="</s>") and (new_tokens[i+1]=="</s>"):
            counter=len(' '.join(raw_strings_1).split(" "))-1
        else:
            pass
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ" or new_tokens[i - 1] == "</s>":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset_1, mention_offset_2    # 表示原始句子中每个单词的索引与编码成token后该单词对应的各个token的索引之间的映射关系


class CoreferenceCrossEncoder(nn.Module):
    def __init__(self, device, args):
        super(CoreferenceCrossEncoder, self).__init__()
        self.device = device
        self.args = args
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = 'CoreferenceCrossEncoder'
        self.mention_model = RobertaModel.from_pretrained('/root/autodl-tmp/PLM/roberta-large',
                                                          return_dict=True)
        self.word_embedding_dim = self.mention_model.embeddings.word_embeddings.embedding_dim  # 1024
        self.mention_dim = self.word_embedding_dim * 2  # 2048
        self.input_dim = int(self.mention_dim * 3)  # 6144
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.mention_model(
            **expected_transformer_input).last_hidden_state
        return transformer_output
    #--------------add---------------#
    def get_sentence_attention(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.mention_model(
            **expected_transformer_input,
            output_attentions=True  # 添加这个参数以获取 attention weights
        )
        attentions = transformer_output.attentions
        return attentions
    #---------------------------------#
    def get_mention_rep(self, transformer_output, start_pieces, end_pieces):
        start_pieces = start_pieces.repeat(1, self.word_embedding_dim).view(
            -1, 1, self.word_embedding_dim)
        start_piece_vec = torch.gather(transformer_output, 1, start_pieces)
        end_piece_vec = torch.gather(
            transformer_output, 1,
            end_pieces.repeat(1, self.word_embedding_dim).view(
                -1, 1, self.word_embedding_dim))
        mention_rep = torch.cat([start_piece_vec, end_piece_vec],
                                dim=2).squeeze(1)  # (10，1，1024*2) -> (10,1024*2)
        return mention_rep  # (10，1024)

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }

    def out(self, combined_rep):
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        return out

    def acc_precision(self, predictions, labels):
        correct = torch.sum(predictions == labels)
        total = float(predictions.shape[0])
        acc = correct / total
        # True Positives (TP): 模型预测为1且标签为1的样本
        true_positives = torch.sum((predictions == 1) & (labels == 1)).float()

        # False Positives (FP): 模型预测为1但标签为0的样本
        false_positives = torch.sum((predictions == 1) & (labels == 0)).float()
        if torch.sum(predictions).item() != 0:
            # precision = torch.sum(
            #     (predictions == labels).float() * predictions == 1) / (
            #         torch.sum(predictions) + sys.float_info.epsilon)
            # Precision (精确率): True Positives / (True Positives + False Positives)
            precision = true_positives / (true_positives + false_positives + sys.float_info.epsilon)
        else:
            precision = torch.tensor(1.0).to(self.device)
        return precision, acc

    def forward(self,
                c_sentences,
                e_sentences,
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels=None):
        transformer_output = self.get_sentence_vecs(sentences)  # 得到最后一层表示，（10，512，1024）
        c_transformer_output = self.get_sentence_vecs(c_sentences)  # 得到最后一层表示，（10，512，1024）
        e_transformer_output = self.get_sentence_vecs(e_sentences)  # 得到最后一层表示，（10，512，1024）
        mention_reps_1 = self.get_mention_rep(transformer_output,
                                              start_pieces_1, end_pieces_1)
        mention_reps_2 = self.get_mention_rep(transformer_output,
                                              start_pieces_2, end_pieces_2)  # (10,1024*2)

        c_mention_reps_1 = self.get_mention_rep(c_transformer_output,
                                              start_pieces_1, end_pieces_1)
        c_mention_reps_2 = self.get_mention_rep(c_transformer_output,
                                              start_pieces_2, end_pieces_2)  # (10,1024*2)


        e_mention_reps_1 = self.get_mention_rep(e_transformer_output,
                                              start_pieces_1, end_pieces_1)
        e_mention_reps_2 = self.get_mention_rep(e_transformer_output,
                                              start_pieces_2, end_pieces_2)  # (10,1024*2)

        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2],
            dim=1)  # (10, 1024*2+1024*2+1024*2)->(10,6144)

        c_combined_rep = torch.cat(
            [c_mention_reps_1, c_mention_reps_2, c_mention_reps_1 * c_mention_reps_2],
            dim=1)  # (10, 1024*2+1024*2+1024*2)->(10,6144)

        e_combined_rep = torch.cat(
            [e_mention_reps_1, e_mention_reps_2, e_mention_reps_1 * e_mention_reps_2],
            dim=1)  # (10, 1024*2+1024*2+1024*2)->(10,6144)

        combined_rep = combined_rep

        f_out = self.out(combined_rep)
        c_out = self.out(c_combined_rep)
        e_out = self.out(e_combined_rep)
        # loss
        cf_out = f_out + torch.tanh(c_out) + torch.tanh(e_out)

        # cf_out = f_out - self.args.alpha*c_out - self.args.beta*e_out
        cf_probs = torch.sigmoid(cf_out)
        f_probs = torch.sigmoid(f_out)
        c_probs = torch.sigmoid(c_out)
        e_probs = torch.sigmoid(e_out)
        probs = cf_probs - e_probs

        # predictions = torch.where(f_probs > 0.5, 1.0, 0.0)
        predictions = torch.where(probs > 0.5, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            precision, acc = self.acc_precision(predictions, labels)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            cf_loss = loss_fct(cf_out, labels)
            c_loss = loss_fct(c_out, labels)
            e_loss = loss_fct(e_out, labels)
            loss = cf_loss + self.args.alpha*c_loss + self.args.beta*e_loss
            output_dict["loss"] = loss
        return output_dict
