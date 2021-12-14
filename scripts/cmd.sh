#! /bin/bash

#conda activate python37
#python data_processing/processingchain3.py --atile ../../data/theia2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output --preview True >> logs/log202112071700.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled19_dates.txt --shf ../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp --preview True >> logs/log202112072125.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2018 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled18_dates.txt --shf ../../data/samples_shapefiles/samples_oso2018_T31TCJ.shp --preview True >> logs/log202112141715.txt
