#!/bin/bash
#SBATCH --job-name=HRnet_C
#SBATCH --output=train_Crossnet_cpu.out
#SBATCH -n 40
#SBATCH --mem=60G
#SBATCH -p wholenodeQ

cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05

source activate torch_cpu

python ../train.py --gpu_ids='-1' --input_nc=7 --netG='Cross' --loadSize=512 --fineSize=1024 --resize_or_crop="scale_width_and_crop" --dataroot='/public/home/win0701/dataset/slices_11_20230308' --name=HR_net2 --label_nc=0 --no_instance --nThreads=0