#! /bin/bash

conda activate python37
python data_processing/processingchain.py --atile ../../data/theia2A_zip_img/subset/2019 --output_dir ../../data/theiaL2A_zip_img/subset/output --preview True >> logs/log202112021435.txt
