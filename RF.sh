#!/bin/bash
python change_detection/classification_viz.py -m RF_model/models/rf_seed_0_case_2.pkl -t ../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.npz -i ../../data/theiaL2A_zip_img/output/2018/2018_GapFilled_Image.tif -o ../../results/RF/classificationmap