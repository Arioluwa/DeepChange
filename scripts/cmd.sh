#! /bin/bash

#conda activate python37
#python data_processing/processingchain3.py --atile ../../data/theia2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output --preview True >> logs/log202112071700.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled19_dates.txt --shf ../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp --preview True >> logs/log202112072125.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2018 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled18_dates.txt --shf ../../data/samples_shapefiles/samples_oso2018_T31TCJ.shp --preview True >> logs/log202112141715.txt

python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/subset --output_dir ../../data/theiaL2A_zip_img/subset/output/ --gfd data_processing/subset_dates_.txt --shf ../../data/samples_shapefiles/samples_oso2019_T31TCJ.shp --preview True >> logs/log202112151000.txt

# for grid
python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2019 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled19_dates.txt --shf ../../data/sample_shapefiles/grided/grided_samples_oso2019_T31TCJ.shp --preview True >> logs/log202201220500.txt
python data_processing/processingchain.py --atile ../../data/theiaL2A_zip_img/2018 --output_dir ../../data/theiaL2A_zip_img/output/ --gfd data_processing/gapfilled18_dates.txt --shf ../../data/sample_shapefiles/grided/grided_samples_oso2018_T31TCJ.shp --preview True >> logs/log202201220600.txt

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
python sample_preparation/readsqlit.py --sq ../../data/theiaL2A_zip_img/output/2018/2018_sample_extract.sqlite --chk 5000 --o ../../data/theiaL2A_zip_img/output/2018/  >> logs/log202201272248.txt

# 2019
python sample_preparation/readsqlit.py --sq ../../data/theiaL2A_zip_img/output/2019/2019_sample_extract.sqlite --chk 5000 --o ../../data/theiaL2A_zip_img/output/2019/  >> logs/log202201272250.txt

# ndvi chart
# python ndvi.py -f ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz -g ../data_processing/gapfilled19_dates.txt -o .
python ndvi2.py -f1 ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -f2 ../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.npz -g ../data_processing/gapfilled19_dates.txt -o .

# RF model

python main.py -f ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -t train_val_eval.txt



#map viz
python classification_viz.py -m ../RF_model/models/rf_model_4.pkl -t ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../../data/theiaL2A_zip_img/output/2018/2018_Image.tif -o .

python classification_viz.py -m ../RF_model/models/rf_model_1.pkl -t ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif -o ../../../results/RF

# change detection (MAD) run in the codebase folder
OTB-7.4.0-Linux64/bin/otbcli_MultivariateAlterationDetector -in1 ../results/RF/2018_rf_model_1_map.tif -in2 ../results/RF/2019_rf_model_1_map.tif -out ../results/RF/change_D/rf_model_1_cd_map.tif -ram 2000

# change detection confusion matrix
OTB-7.4.0-Linux64/bin/otbcli_ComputeConfusionMatrix -in ../results/RF/2018_rf_model_1_map.tif -out ../results/RF/change_D/rf_model_1_cm.csv -format confusionmatrix -ref raster -ref.raster.in ../results/RF/2019_rf_model_1_map.tif -ram 2000

# similarity measure
similarity_measure.py -s '../../../results/RF/simliarity_measure/2018_rf_model_2.npy' -t '../../../results/RF/simliarity_measure/2019_rf_model_2.npy' - i '../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif' -o '../../../results/RF/simliarity_measure/'
python similarity_mesure.py -s '../../../results/RF/simliarity_measure/2018_rf_model_2.npy' -t '../../../results/RF/simliarity_measure/2019_rf_model_2.npy' -i '../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif' -o '../../../results/RF/simliarity_measure' -gs '../../../data/rasterized_samples/2018_rasterizedImage.tif' -gt '../../../data/rasterized_samples/2019_rasterizedImage.tif'

# LTAE train 
python train.py --dates dates.txt --positions bespoke --seed 4 
python train.py --dates dates.txt --positions bespoke --seed 4 

# predict
# python predict.py -m ../../../results/ltae/trials/Seed_0/model.pth.tar -r ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilld_Image.tif -o ../../../results/ltae -f 2 -c '../../../results/ltae/trials/conf.json'
# predict RF
python predict.py -m ../RF_model/models/rf_seed_0_case_1.pkl -r ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif -o ../../../results/RF/classificationmap -f 1 -c 1


# train
python train.py --npy ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --epoch 10 --seed 0 --res_dir ../../../results/ltae/results/2018 --num_workers 10 --batch_size 128 --lr 0.01 --positions bespoke --_scheduler True

C#ategory H2
python train.py --npy ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --epoch 30 --seed 0 --res_dir ../../../results/ltae/results/2018 --num_workers 10 --batch_size 2048 --factor 5266 --positions Bespoke --scheduler_ True >>../logs/log20220429.txt

python train.py --npy ../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz --epoch 15 --seed 0 --res_dir ../../../results/ltae/results/2018 --num_workers 10 --batch_size 2048 --factor 5266 --positions Bespoke --scheduler_ True >>../logs/log202205012326.txt

# optimal threshold 
python optimal_threshold.py -o ../../../results/ltae/Change_detection/similarity_measure 
python optimal_threshold.py -o ../../../results/ltae/Change_detection/similarity_measure -ot True

