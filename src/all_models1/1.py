# from DualGCN_crossencoder_trainer import *
# sent_id = 70
# tri_id = 44
# target_sent1 = None
# target_sent2 = None
# train_df = pd.read_csv(config_dict["train_path"], index_col=0)
# sentences_text_1 = train_df['text_1'][sent_id]
# sentences_text_2 = train_df['text_2'][sent_id]
# trigger_1_abs_start = train_df['trigger_1_abs_start'][tri_id]
# trigger_1_abs_end = train_df['trigger_1_abs_end'][tri_id]
# trigger_2_abs_start = train_df['trigger_2_abs_start'][tri_id]
# trigger_2_abs_end = train_df['trigger_2_abs_end'][tri_id]
#
# def sent(trigger_1_abs_start, sentences_text_1):
#     import nltk
#
#     text = sentences_text_1
#
#     sentences = nltk.sent_tokenize(text)
#     print(sentences)
#     # 给定索引位置
#     word_index = trigger_1_abs_start
#
#     # 迭代句子以找到索引位置
#     current_index = 0
#     target_sentence = None
#     count = 0
#
#     for sentence in sentences:
#         # if count == 68:
#         #     print('1')
#         #     print(sentence)
#         if sentence is None:
#             continue  # 如果句子为 None，跳过
#
#         sentence_length = len(sentence)
#         if current_index <= word_index < current_index + sentence_length:
#             target_sentence = sentence
#             break
#         current_index += sentence_length + 1  # 加1是因为句子之间有空格分隔
#         count += 1
#
#
#     print(target_sentence)
#     return target_sentence
#
# if __name__ == '__main__':
#     print('第一个句子：')
#     target_sent1 = sent(trigger_1_abs_start, sentences_text_1)
#     print('第二个句子：')
#     target_sent2 = sent(trigger_2_abs_start, sentences_text_2)
#     print('第一个和第二个句子')
#     print(target_sent1 +' '+ target_sent2)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import pickle
from DualGCN_crossencoder_trainer import *
# # 打开文件
# with open('/home/yaolong/Rationale4CDECR-main/data_preparation/data_set_in_tensor_no_pad.pkl', 'rb') as file:
#     # 使用 pickle.load() 加载文件中的对象
#     loaded_data = pickle.load(file)
#
# def pad_to_size(array, target_size):
#     padded_array = np.pad(array, (0, target_size - array.shape[0]), 'constant')
#     return padded_array
#
# data_set_in_tensor = loaded_data['data_set_in_tensor']
# all_syn_adj = loaded_data['all_syn_adj']
#
# sentence = data_set_in_tensor[0]
# start_pieces_1 = data_set_in_tensor[1]
# end_pieces_1 = data_set_in_tensor[2]
# start_pieces_2 = data_set_in_tensor[3]
# end_pieces_2 = data_set_in_tensor[4]
# labels = data_set_in_tensor[5]
#
# target_size = 512
# padded_arrays = [pad_to_size(arr, target_size) for arr in all_syn_adj]
# # 创建新的 TensorDataset
# # padded_arrays_tensor = torch.tensor(padded_arrays)
#
# from torch.utils.data import Dataset, DataLoader
# class PairedDataset(Dataset):
#     def __init__(self, data_set_in_tensor, all_syn_adj):
#         self.data_set_in_tensor = data_set_in_tensor
#         self.all_syn_adj = all_syn_adj
#
#     def __len__(self):
#         # 假设两个张量的长度相同
#         return len(self.all_syn_adj)
#
#     def __getitem__(self, idx):
#         # 返回两个张量中对应的数据对
#         data = self.data_set_in_tensor[idx]
#         syn_adj = self.all_syn_adj[idx]
#         return (*data, syn_adj)
#
#
# batch_size = 10  # 根据内存大小调整批量大小
#
# # 创建 PairedDataset 实例
# paired_dataset = PairedDataset(data_set_in_tensor, padded_arrays)
#
# # 创建 DataLoader 实例
# data_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
# batcher = tqdm(data_loader, desc="Batch")
# for step, batch in enumerate(batcher):
#     *data_batch, syn_adj_batch = batch
# print('ok')

with open('/home/yaolong/Rationale4CDECR-main/data_preparation/dev_data_output.pkl', 'rb') as file:
    # 使用 pickle.load() 加载文件中的对象
    loaded_data = pickle.load(file)

print('ok')