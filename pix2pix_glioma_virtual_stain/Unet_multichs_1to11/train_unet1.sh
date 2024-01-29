#!/bin/bash
#SBATCH --job-name=nor_U1
#SBATCH --output=train_unet1.out
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

python ../train.py --input_nc=1 --netG='UNet' --loadSize=512 --fineSize=512 --resize_or_crop="scale_width_and_crop" --label_nc=0 --no_instance --nThreads=0 --dataroot_A=/public/home/win0704/pix2pixHD/Datasets_match_6/data_11/trainA_sum_11 --dataroot_B=/public/home/win0704/pix2pixHD/Datasets_match_6/data_11/trainB_normalization --dataroot_custom --datatype_mat_n=1 --name=Unet1