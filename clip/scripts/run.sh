#!/bin/bash
#SBATCH -A sian
#SBATCH --job-name clip
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=u22
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sian.shinjo@students.iiit.ac.in
#SBATCH --output=/home2/sian/summer-project/clip/logs/%j.out
#SBATCH --error=/home2/sian/summer-project/clip/logs/%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=gnode085

# u for undefined vars and pipefails
set -uo pipefail
# prints every cmd for debugging
set -x

# module load u22/cuda/11.8

echo "Job started on: $(hostname)"
echo "My GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "My CPUs: $(nproc)"

mkdir -p logs checkpoints

SCRATCH_DIR="/ssd_scratch/cvit/sian/datasets/Flickr8k"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

if [ ! -d "images" ]; then
    echo "Downloading Flickr8k directly to ssd_scratch..."
    # Download images and text
    wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
    
    echo "Unzipping..."
    unzip -q Flickr8k_Dataset.zip -d images
    unzip -q Flickr8k_text.zip -d text
    
    # Clean up zips to save SSD space
    rm Flickr8k_Dataset.zip Flickr8k_text.zip
else
    echo "Data already exists on this node's scratch. Skipping download."
fi

cd /home2/sian/summer-project/clip

# force offline mode to prevent wandb from hanging the cluster
# export WANDB_MODE=offline

echo "Launching toy CLIP training on single GPU..."

/home2/sian/summer-project/vision_transformer/.venv/bin/python -u train.py --config config/toy-clip.json

echo "Training completed successfully."
