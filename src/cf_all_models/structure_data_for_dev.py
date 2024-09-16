from structure_data_for_train_and_dev import *


dev_event_pairs, dev_pairs, dev_docs = structure_dataset_for_eval(dev_data, eval_set='dev')
# 将三个输出保存为一个字典
output_data = {
    'dev_event_pairs': dev_event_pairs,
    'dev_pairs': dev_pairs,
    'dev_docs': dev_docs
}

# 保存字典到一个 .pkl 文件中
print("dev保存为文件...")
with open('/home/yaolong/Rationale4CDECR-main/data_preparation/dev_data_output.pkl', 'wb') as f:
    pickle.dump(output_data, f)

print("dev保存成功...")