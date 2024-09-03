#!/bin/bash -l

#SBATCH -J mvLibri_baseline  #name of the job 
#SBATCH --partition=gpu-all
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100nv_32GB:2
#SBATCH --time=96:00:00
#SBATCH --error=job.%J.err
##SBATCH --output=job.%J.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
source path.sh
bash run_mv_km2000_bpe_rm6k_ts5k.sh


