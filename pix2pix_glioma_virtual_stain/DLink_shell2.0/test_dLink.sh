#!/bin/bash
#SBATCH --job-name=test_Dlink
#SBATCH --output=test_Dlink.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH -t 168:0:0
#SBATCH -p gpu
#SBATCH --gres=gpu:1              # 每个节点上申请一块GPU卡


cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05
module load compiler/cuda/11.0-with-cuDNN8.2.1

source activate pytorch 

python ../test.py --no_resie_crop --input_nc=63 --loadSize=2048 --fineSize=1024 --resize_or_crop="scale_width_and_crop" --netG="DinkNet34" --dataroot='/public/home/win0701/dataset/npy63ch_nml&can' --results_dir='../results/D_Link' --checkpoints_dir='../checkpoints/D_Link' --label_nc=0 --no_instance --nThreads=0 --use_encoded_image --serial_batches 