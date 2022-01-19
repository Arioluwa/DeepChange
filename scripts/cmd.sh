#! /bin/bash

#conda activate python37
#python data_processing/processingchain3.py --atile ../../data/theia2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output --preview True >> logs/log202112071700.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled19_dates.txt --shf ../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp --preview True >> logs/log202112072125.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2018 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled18_dates.txt --shf ../../data/samples_shapefiles/samples_oso2018_T31TCJ.shp --preview True >> logs/log202112141715.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/subset --output_dir ../../data/theiaL2A_zip_img/subset/output/ --gfd data_processing/subset_dates_.txt --shf ../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp --preview True >> logs/log202112151000.txt

# for grid
python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled19_dates.txt --shf ../../data/samples_shapefiles/grided/grided_samples_oso2019_T31TCJ.shp --preview True >> logs/log202201200018.txt
python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2018 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled18_dates.txt --shf ../../data/samples_shapefiles/grided/grided_samples_oso2018_T31TCJ.shp --preview True >> logs/log202201200018.txt

# Rasterization
# To be run in /OTB-7.4.0-Linux64/bin
./otbcli_Rasterization -in ../../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp -out ../../../data/sample_rasterized/2019_rasterizedImage.tif -im ../../../data/theiaL2A_zip_img/output/2019/2019_Mask.tif -mode attribute -mode.attribute.field code -ram 2000 >> ../../DeepChange/logs/log202201071311.txt

# Confusion matrix
./otbcli_ComputeConfusionMatrix -in ../../../data/sample_rasterized/2018_rasterizedImage.tif -out ../../../data/sample_rasterized/sample_confusionMatrix.csv -format confusionmatrix -ref raster -ref.raster.in ../../../data/sample_rasterized/2019_rasterizedImage.tif -ram 2000 >> ../../DeepChange/logs/log202201071427.txt

../../../data/theiaL2A_zip_img/output/2018/2018_sample_extract.sqlite

# to be run in sample preparation folder
# npz with initialization matrix
python npz_init.py --sq ../../../data/theiaL2A_zip_img/output/2018/2018_sample_extract.sqlite --chk 5000 --o ../../../data/theiaL2A_zip_img/output/2018/ --d ../data_processing/gapfilled19_dates >> ../logs/log202201092254.txt

# 2018
python readsqlit2.py --sq ../../../data/theiaL2A_zip_img/output/2018/2018_sample_extract.sqlite --chk 5000 --o ../../../data/theiaL2A_zip_img/output/2018/  >> ../logs/log202201092254.txt

# 2019
python readsqlit2.py --sq ../../../data/theiaL2A_zip_img/output/2019/2019_sample_extract.sqlite --chk 5000 --o ../../../data/theiaL2A_zip_img/output/2019/  >> ../logs/log202201092345.txt