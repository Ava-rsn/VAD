#!/bin/bash
#SBATCH --job-name=fast_postproc_shanghai
#SBATCH --output=logs/fast_postproc_shanghai_%j.out
#SBATCH --error=logs/fast_postproc_shanghai_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=coms037985


set -euo pipefail

mkdir -p logs

source ~/venvs/aedmae/bin/activate

cd ~/evaluations

python range_search.py \
  --data_dir ~/evaluations/shanghai_shanghai_files \
  --dataset_prefix shanghai \
  --score_types teacher,combined,st \
  --ranges 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500 \
  --mus 50,100,150,200,250,300,350,400,450,550,600 \
  --normalizes true,false \
  --out_csv shanghai_fast_search_results.csv \
  --out_txt shanghai_fast_search_summary.txt