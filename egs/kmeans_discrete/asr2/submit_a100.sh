#!/bin/bash -l

#SBATCH -J repcodec
#SBATCH -o job.%J.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:A100_80GB:1
#SBATCH --mem 80GB
eval "$(conda shell.bash hook)"
. /export/home/vsukhadia/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate espnet

#srun run_mv.sh
bash run.sh
