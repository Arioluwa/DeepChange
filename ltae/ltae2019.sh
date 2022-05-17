#!/bin/bash -l
#SBATCH --chdir=/share/projects/erasmus/deepchange/codebase/DeepChange/ltae
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --partition shortrun
#SBATCH --output 2019_outputs.out
#SBATCH --time=2-00:00:00


setcuda 11.0
conda activate python_env

### OPTIONAL, copy data project data ###
# python predict.py -m ../../../results/ltae/trials/Seed_0/model.pth.tar -t ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif -o ../../../results/ltae/results -f 2 -c '../../../results/ltae/trials/conf.json'
# python train.py --npy ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --epoch 100 --seed 2 --num_workers 4 --batch_size 2048
# python train.py --npy ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz --factor 1300 --epoch 100 --seed 2 --res_dir ../../../results/ltae/results/2019 --num_workers 4 --batch_size 2048 --positions bespoke 
python alt-train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2019 --res_dir ../../../results/ltae/model/2019 --epochs 15 --seed 1; python alt-train.py --dataset_folder ../../../data/theiaL2A_zip_img/output/2019 --res_dir ../../../results/ltae/model/2019 --epochs 15 --seed 2
