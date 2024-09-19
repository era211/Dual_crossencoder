DATE=`date '+%Y-%m-%d-%H:%M:%S'`

gpu_num=0
random_seed=5
title=main
dataset=ecb # `fcc' or `gvc'
exp_name=baseline # or `enhanced'
config_path=configs/${title}/${dataset}/${exp_name}.json
out_dir=outputs/${title}/${dataset}/cf/train_test/${exp_name}/best_model
out_dir_test=outputs/${title}/${dataset}/cf/train_test/${exp_name}/eval_results
if [ ! -d "$out_dir" ];then
mkdir -p $out_dir
fi

# train crossencoder
echo "Train crossencoder"

nohup python -u src/cf_all_models/crossencoder_trainer_all.py --config_path ${config_path} --out_dir ${out_dir} --out_dir_test ${out_dir_test} \
    --mode train --random_seed ${random_seed} --gpu_num ${gpu_num} >${out_dir}/crossencoder.log 2>${out_dir}/crossencoder.progress &
