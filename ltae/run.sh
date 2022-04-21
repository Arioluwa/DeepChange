#!/bin/bash -l
#SBATCH --chdir=/share/projects/erasmus/deepchange/codebase/DeepChange/ltae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --partition shortrun
#SBATCH --output my_output.out
#SBATCH --time=2-00:00:00


setcuda 11.0
conda activate python37

### OPTIONAL, copy data project data ###

python train.py --npy ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --epoch 100 --seed 2 --num_workers 4 --batch_size 2048