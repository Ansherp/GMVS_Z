#!/bin/bash
#SBATCH --job-name=global
#SBATCH --output=test_global.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40G
#SBATCH -t 168:0:0
#SBATCH -p gpu
#SBATCH --gres=gpu:1              # 每个节点上申请一块GPU卡


cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05
module load compiler/cuda/11.0-with-cuDNN8.2.1

source activate pytorch

python ../test.py --input_nc=1 --netG='global' --loadSize=512 --fineSize=512 --resize_or_crop="scale_width_and_crop" --dataroot='/public/home/win0704/pix2pixHD/Datasets_match_6' --name=Global --label_nc=0 --no_instance --nThreads=0 --use_encoded_image --which_epoch=120
