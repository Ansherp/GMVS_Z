#!/bin/bash
#SBATCH --job-name=HRnet_C
#SBATCH --output=train_HRnet_cpu.out
#SBATCH -n 40
#SBATCH --mem=60G
#SBATCH -p wholenodeQ

cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05

source activate torch_cpu

python ../train.py --gpu_ids='-1' --input_nc=63 --netG='HRnet' --loadSize=2048 --fineSize=1024 --resize_or_crop="scale_width_and_crop" --dataroot='/public/home/win0701/dataset/npy63ch_nml&can/' --checkpoints_dir='../checkpoints/HRnet_cpu/' --label_nc=0 --no_instance --nThreads=0