DATE=`date '+%Y-%m-%d-%H:%M:%S'`

gpu_num=0
random_seed=5
title=main
dataset=ecb # `fcc' or `gvc'
exp_name=baseline # or `enhanced'
losstype=differentiatedloss # 'doubleloss', 'orthogonalloss', 'differentiatedloss'
load_data=True # True
save_data=False # False
dual=dual
config_path=configs/${title}/${dataset}/${exp_name}.json
out_dir=outputs/${title}/${dataset}/${exp_name}/${dual}/dual_save_data/best_model

if [ ! -d "$out_dir" ];then
mkdir -p $out_dir
fi

# train crossencoder
echo "Train dual_crossencoder"

nohup python -u src/all_models/DualGCN_crossencoder_v4.py --config_path ${config_path} --out_dir ${out_dir}\
    --mode train --load_data True --save_data ${save_data} --losstype differentiatedloss --alpha 0.4 --beta 0.3 --random_seed ${random_seed} --gpu_num ${gpu_num} >${out_dir}/dual_crossencoder_v4_savedata.log 2>${out_dir}/dual_crossencoder_v4_savedata.progress &
