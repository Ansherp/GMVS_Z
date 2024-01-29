#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --nodes=1                 # 申请一个节点
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 每个节点上申请一块GPU卡

cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05
module load compiler/cuda/11.0-with-cuDNN8.2.1

source activate pytorch

python train.py --print_freq=10 --niter=100 --niter_decay=10 --input_nc=3 --loadSize=512 --fineSize=512 --resize_or_crop="scale_width_and_crop" --dataroot='datasets' --checkpoints_dir='checkpoints/test' --label_nc=0 --no_instance --nThreads=0 --serial_batches