#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=10:00:00 
#SBATCH --gres=gpu:1

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 'google/mt5-base' \
    --do_train \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/counterfactual_question_have-can_de/counterfactual_question_have_de.train.json \
    --validation_file ../data/counterfactual_question_have-can_de/counterfactual_question_have_de.dev.json \
    --output_dir $SCRATCH/mt5-counterfactual-question-have-de/	 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 1.0
"