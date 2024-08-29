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
    return embeddings, mapping, mention_offset_1, mention_offset_2, mention_sent_1, mention_sent_2    # 表示原始句子中每个单词的索引与编码成token后该单词对应的各个token的索引之间的映射关系


class CoreferenceCrossEncoder(nn.Module):
    def __init__(self, device):
        super(CoreferenceCrossEncoder, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = 'CoreferenceCrossEncoder'
        self.mention_model = RobertaModel.from_pretrained('/home/yaolong/PT_MODELS/PT_MODELS/roberta-large',
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

    def forward(self,
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels=None):
        transformer_output = self.get_sentence_vecs(sentences)  # 得到最后一层表示，（10，512，1024）
        mention_reps_1 = self.get_mention_rep(transformer_output,
                                              start_pieces_1, end_pieces_1)
        mention_reps_2 = self.get_mention_rep(transformer_output,
                                              start_pieces_2, end_pieces_2)  # (10,1024*2)
        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2],
            dim=1)  # (10, 1024*2+1024*2+1024*2)->(10,6144)
        combined_rep = combined_rep
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > 0.5, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            correct = torch.sum(predictions == labels)
            total = float(predictions.shape[0])
            acc = correct / total
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * predictions == 1) / (
                        torch.sum(predictions) + sys.float_info.epsilon)
            else:
                precision = torch.tensor(1.0).to(self.device)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            loss = loss_fct(out, labels)
            output_dict["loss"] = loss
        return output_dict



class CoreferenceCrossEncoder_DualGCN(nn.Module):
    def __init__(self, device, args):
        super(CoreferenceCrossEncoder_DualGCN, self).__init__()
        self.device = device
        self.args = args
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = 'CoreferenceCrossEncoder_DualGCN'
        self.mention_model = RobertaModel.from_pretrained('/home/yaolong/PT_MODELS/PT_MODELS/roberta-large',
                                                          return_dict=True).to(device)
        self.gcn_model = GCNModel(device, self.mention_model, opt=args).to(device)
        self.word_embedding_dim = self.mention_model.embeddings.word_embeddings.embedding_dim  # 1024
        self.mention_dim = self.word_embedding_dim * 2  # 2048
        self.input_dim = int(self.mention_dim * 4)  # 8192
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_0 = nn.Linear(self.input_dim, self.input_dim - 2048)  # (8192-->6144)
        self.hidden_layer_1 = nn.Linear(self.input_dim - 2048, self.mention_dim)  # (6144-->2048)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)  # (2048-->2048)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)  # (2048-->1)

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
            -1, 1, self.word_embedding_dim)  # (10,1,1024)
        start_piece_vec = torch.gather(transformer_output, 1, start_pieces)
        end_piece_vec = torch.gather(
            transformer_output, 1,
            end_pieces.repeat(1, self.word_embedding_dim).view(
                -1, 1, self.word_embedding_dim))  # (10,1,1024)
        mention_rep = torch.cat([start_piece_vec, end_piece_vec],
                                dim=2).squeeze(1)  # (10，1，1024*2) -> (10,1024*2)
        return mention_rep  # (10，1024*2)

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }


    def peanl(self, adj_ag, adj_dep):
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).to(self.device)
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).to(self.device)

        penal = None
        if self.args.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).to(self.device)
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(self.device)
            penal = self.args.alpha * penal1 + self.args.beta * penal2

        elif self.args.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).to(self.device)
            penal = self.args.alpha * penal

        elif self.args.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(self.device)
            penal = self.args.beta * penal

        return penal

    def forward(self,
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                adj1,
                adj2,
                all_embeddings_ment1,
                all_embeddings_ment2,
                labels=None):
        transformer_output = self.get_sentence_vecs(sentences)  # 得到最后一层表示，（10，512，1024）

        transformer_output_ment1 = self.get_sentence_vecs(all_embeddings_ment1)  # 得到最后一层表示，（10，100，1024）
        transformer_output_ment2 = self.get_sentence_vecs(all_embeddings_ment2)

        mention_reps_1 = self.get_mention_rep(transformer_output,
                                              start_pieces_1, end_pieces_1)  # (10,2048)  # 得到事件触发词的特征向量
        mention_reps_2 = self.get_mention_rep(transformer_output,
                                              start_pieces_2, end_pieces_2)  # (10,1024*2)

        outputs1_1, outputs1_2, outputs2_1, outputs2_2, adj_ag_1, adj_ag_2, adj1, adj2 = self.gcn_model(transformer_output_ment1, transformer_output_ment2, start_pieces_1, end_pieces_1,
                                                                            start_pieces_2, end_pieces_2, adj1, adj2,
                                                                            all_embeddings_ment1, all_embeddings_ment2)

        # outputs1:(10,512)
        combined_rep = torch.cat(
            [outputs1_1, outputs1_2, outputs2_1, outputs2_2, mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2],
            dim=1)  # (10, 512*4+2048*3)
        combined_rep = combined_rep
        zero_hidden = F.relu(self.hidden_layer_0(combined_rep))
        first_hidden = F.relu(self.hidden_layer_1(zero_hidden))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > 0.5, 1.0, 0.0)  # 预测值1或0
        output_dict = {"probabilities": probs, "predictions": predictions}  # 包含概率和预测的0、1的值
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            correct = torch.sum(predictions == labels)  # 这一个batch中预测正确的概率
            total = float(predictions.shape[0])  # 一个batch中样本总数
            acc = correct / total  # 准确率，准确率变低说明正确的个数变少了
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * (predictions == 1)) / (
                        torch.sum(predictions) + sys.float_info.epsilon)

                TP = torch.sum((predictions == 1) & (labels == 1)).float()
                FP = torch.sum((predictions == 1) & (labels == 0)).float()
                FN = torch.sum((predictions == 0) & (labels == 1)).float()
                precision1 = TP / (TP + FP + sys.float_info.epsilon)
                recall = TP / (TP + FN + sys.float_info.epsilon)
                f1_score = 2 * (precision1 * recall) / (precision1 + recall + sys.float_info.epsilon)

            else:
                recall = 0.0
                f1_score = 0.0
                precision = torch.tensor(1.0).to(self.device)

            penal1 = self.penal(adj_ag_1, adj1)
            penal2 = self.penal(adj_ag_2, adj2)

            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            output_dict["recall"] = recall
            output_dict["f1_score"] = f1_score
            loss = loss_fct(out, labels) + self.args.penal_alpha*penal1 + self.args.penal_beta*penal2
            output_dict["loss"] = loss


        return output_dict


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class GCNModel(nn.Module):
    def __init__(self, device, bert, opt):
        super().__init__()
        self.opt = opt
        self.device = device
        self.gcn = GCNBert(device, bert, opt, opt.num_layers).to(device)

    def avg_pooling(self, h1, h2, sent):
        mask = sent != 1  # (10, 100)句子部分为1，padding部分为0
        mask = torch.tensor(mask, dtype=torch.bool)
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # (10, 1)
        aspect_mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)  # (10, 100, 512)
        outputs1 = (h1 * aspect_mask).sum(dim=1) / asp_wn  # (10, 512)
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2

    def forward(self, transformer_output_ment1, transformer_output_ment2, start_pieces_1, end_pieces_1,
                                                start_pieces_2, end_pieces_2, adj1, adj2,
                                                all_embeddings_ment1, all_embeddings_ment2):
        h1_1, h2_1, adj_ag_1, pooled_output_1 = self.gcn(transformer_output_ment1, adj1, all_embeddings_ment1)
        h1_2, h2_2, adj_ag_2, pooled_output_2 = self.gcn(transformer_output_ment2, adj2, all_embeddings_ment2)


        # avg pooling asp feature
        outputs1_1, outputs1_2 = self.avg_pooling(h1_1, h2_1, transformer_output_ment1)
        outputs2_1, outputs2_2 = self.avg_pooling(h1_2, h2_2, transformer_output_ment2)

        return outputs1_1, outputs1_2, outputs2_1, outputs2_2, adj_ag_1, adj_ag_2, adj1, adj2


class GCNBert(nn.Module):
    def __init__(self, device, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.device = device
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim).to(self.device)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
        transformer_output = self.bert(
            **expected_transformer_input).last_hidden_state
        return transformer_output

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }

    def forward(self, transformer_output_ment, adj, all_embeddings_ment):
        mask = all_embeddings_ment != 1  # 句子部分为1，padding部分为0
        src_mask = mask.unsqueeze(-2) # (10, 100)-->(10, 1, 100)
        sequence_output = self.layernorm(transformer_output_ment)  # (10，100，1024)对bert输出进行层归一化
        gcn_inputs = self.bert_drop(sequence_output)  # (10, 100, 1024)将归一化后的输出进行dropout操作，然后作为后续图神经网络的输入
        pooled_output = transformer_output_ment[:, 0, :]  # (10, 1024)
        pooled_output = self.pooled_drop(pooled_output)  # 对输出的cls编码进行dropout操作

        denom_dep = adj.sum(2).unsqueeze(2) + 1  # (10, 100, 1)用于语法图卷积
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)   # (10,1,100,100) 将输入进行多头注意力操作
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]  # list:(10, 100,100)
        multi_head_list = []
        outputs_dep = None
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]  # (10, 100, 100)
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] = adj_ag[j] - torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] = adj_ag[j] + torch.eye(adj_ag[j].size(0)).to(self.device)
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1  # (10, 100 ,1)
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs
        adj = adj.to(torch.float32)

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = adj.bmm(outputs_dep)  # adj:(10,100,100), outputs_dep:(10, 100, 1024)
            AxW_dep = self.W[l](Ax_dep)  # (10, 100, 512)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep).to(torch.float32)  # (10, 100, 512)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag, pooled_output


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)  # 1024
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # (10,1,512,512)

    p_attn = F.softmax(scores, dim=-1)  # (10,1,512,512)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]  # (10,1,512)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (10,1,1,512)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)  # (10,1,512,512)
        return attn