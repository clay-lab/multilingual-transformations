#!/bin/bash

#SBATCH --job-name=MT5-base-eval-neg-de
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=07:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL

module load CUDA
module load cuDNN
module load miniconda

source activate /gpfs/loomis/project/frank/ref4/conda_envs/py38

python models/run_seq2seq.py \
	--model_name_or_path 'google/mt5-base' \
	--do_eval \
	--do_learning_curve \
	--task translation_src_to_tgt \
	--train_file data/neg_de/neg_de_train.json.gz \
	--validation_file data/neg_de-no_indef/neg_de-no_indef_test.json.gz \
	--output_dir outputs/mt5-finetuning-neg-de-bs128/ \
	--per_testice_train_batch_size=4 \
	--per_testice_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \
