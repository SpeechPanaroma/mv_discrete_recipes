#!/bin/bash -l

#SBATCH -J repcodec
#SBATCH -o job.%J.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:A100_80GB:2
#SBATCH --mem 80GB
eval "$(conda shell.bash hook)"
. /alt-asr/shchowdhury/tools/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate espnet_310_202310
bash run_mv_km2000_bpe_rm6k_ts5k.sh
