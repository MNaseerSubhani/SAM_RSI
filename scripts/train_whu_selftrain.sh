#!/bin/bash

cfg_file="configs.config_whu"
prompt="point"
load_type="soft"
num_points_list=(1 2 3)
output_dirs=("work_dir/whu/selftrain")

for output_dir in "${output_dirs[@]}"; do
    for num_points in "${num_points_list[@]}"; do
        out_dir="${output_dir}/point_${num_points}"
        CUDA_VISIBLE_DEVICES=0 python train_selftrain.py --cfg "$cfg_file" --prompt "$prompt" --num_points "$num_points" --out_dir "$out_dir" --load_type "$load_type"
    done
done
