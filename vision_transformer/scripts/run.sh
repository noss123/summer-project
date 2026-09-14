#!/bin/bash
#SBATCH -A sian
#SBATCH --job-name vision-transformer
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=u22
#SBATCH --time=0-08:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sian.shinjo@students.iiit.ac.in
#SBATCH --output=/home2/sian/summer-project/vision-transformer/logs/%j.out
#SBATCH --error=/home2/sian/summer-project/vision-transformer/logs/%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=gnode083

# u for undefined vars and pipefails
set -uo pipefail
# prints every cmd for debugging
set -x

module load u22/cuda/11.8

source /home2/sian/summer-project/vision-transformer/.venv/bin/activate

echo "Job started on: $(hostname)"
echo "My GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi
echo "My CPUs: $(nproc)"

# RUN 1: BASELINE ViT
echo "Launching Baseline study on GPUs 0 and 1..."
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=44567 \
    train.py --config config/baseline.json \
    > logs/baseline_run.log 2>&1 &

# RUN 2: ViT WITH CUTMIX
echo "Launching CutMix study on GPUs 2 and 3..."
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --master_port=44568 \
    train.py --config config/cutmix.json \
    > logs/cutmix_run.log 2>&1 &

# synchronise by waiting for both runs to finish
wait

echo "All runs completed successfully."
