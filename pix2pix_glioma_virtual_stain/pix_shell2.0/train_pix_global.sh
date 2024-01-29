#!/bin/bash
#SBATCH --job-name=p_global
#SBATCH --output=train_global.out
#SBATCH --nodes=1                 # 申请一个节点
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 每个节点上申请一块GPU卡

cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05
module load compiler/cuda/11.0-with-cuDNN8.2.1

source activate pytorch 

python ../train.py --input_nc=1 --loadSize=512 --fineSize=512 --resize_or_crop="scale_width_and_crop" --dataroot='/public/home/win0704/pix2pixHD/Datasets_match_6' --name=Global --label_nc=0 --no_instance --nThreads=0 --serial_batches