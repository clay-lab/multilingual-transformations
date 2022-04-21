#!/bin/bash

#SBATCH --job-name=MT5-base-eval-neg-en
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB 
#SBATCH --time=02:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu

module load CUDA
module load cuDNN
module load miniconda

source activate /gpfs/loomis/project/frank/ref4/conda_envs/py38

python models/run_seq2seq.py \
    --model_name_or_path 'google/mt5-base' \
    --do_eval \
    --do_learning_curve \
    --task translation_src_to_tgt \
    --train_file data/neg_en/neg_en_train.json.gz \
    --validation_file data/neg_en/neg_en_dev.json.gz \
    --output_dir outputs/mt5-finetuning-neg-en-bs128/  \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
