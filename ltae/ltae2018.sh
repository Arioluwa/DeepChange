#!/bin/bash -l
#SBATCH --chdir=/share/projects/erasmus/deepchange/codebase/DeepChange/ltae
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition shortrun
#SBATCH --output 2018_output.out
#SBATCH --time=2-00:00:00


setcuda 11.0
conda activate python_env

### OPTIONAL, copy data project data ###

python alt-train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2018 --res_dir ../../../results/ltae/model/2018/second --epochs 15; python alt-train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2018 --res_dir ../../../results/ltae/model/2018/third --epochs 15; python alt-train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2018 --res_dir ../../../results/ltae/model/2018/fourth --epochs 15