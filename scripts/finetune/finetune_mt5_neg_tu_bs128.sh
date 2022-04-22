#!/bin/bash

#SBATCH --job-name=MT5-base-finetune-neg-tu
#SBATCH --output=joblogs/test_neg_tu_%j.txt
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB 
#SBATCH --time=07:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu

module load miniconda

source activate py38

cd ~/project/multilingual_transformations

python models/run_seq2seq.py \
    --model_name_or_path 'google/mt5-base' \
    --do_train \
    --task translation_src_to_tgt \
    --train_file data/neg_tu/neg_tu_train.json.gz \
    --validation_file data/neg_tu/neg_tu_dev.json.gz \
    --output_dir outputs/mt5-finetuning-neg-tu-bs128/  \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=32 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 10.0
