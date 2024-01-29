#!/bin/bash
#SBATCH --job-name=test_HR_C
#SBATCH --output=test_HRnet_cpu.out
#SBATCH -n 40
#SBATCH --mem=100G
#SBATCH -p wholenodeQ

cd $SLURM_SUBMIT_DIR

module load apps/anaconda3/2021.05

source activate torch_cpu

python ../test.py --no_resie_crop --gpu_ids='-1' --data_type=32 --input_nc=63 --loadSize=2048 --fineSize=1024 --resize_or_crop="scale_width_and_crop" --netG="HRnet" --dataroot='/public/home/win0701/dataset/npy63ch_nml&can' --results_dir='../results/HRnet_cpu' --checkpoints_dir='../checkpoints/HRnet_cpu' --label_nc=0 --no_instance --nThreads=0 --use_encoded_image --serial_batches