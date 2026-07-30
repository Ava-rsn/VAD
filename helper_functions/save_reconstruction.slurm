#!/bin/bash
#SBATCH --job-name=avenue_inference
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=Avenue_avenue_inference_logs/%x_%j.out
#SBATCH --error=Avenue_avenue_inference_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --account=coms037985

set -euo pipefail

mkdir -p Avenue_avenue_inference_logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi || true

module purge
module load cuda/11.8.0-rgxs

source ~/venvs/aedmae/bin/activate

cd ~/evaluations

python save_reconstructions_ucf.py --dataset ucf_crime --video_name Assault042_x264 --frame_idx 30 --output_dir checkpoints/shanghai --save_dir recon_vis/Assault042/normal