#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import sqlite3
import os
import argparse
import time

# =================
# Data processing and preparation
# From the sqlite file generate from the processing chain script, this script reads the sqlite file in chunks extracts the X (features vectors), y (label), block_id  (assigned random number for each block (splitted grid) 0-100, added to each sample polygon before running the processing chain), and the polygon_id.
#
# Input: Sqlite file
# It output a npz file, for easy read into the models and later use
# =================

parser = argparse.ArgumentParser(description='Read SITS data')

parser.add_argument('--sq', '--sqlitepath',type=str, help='path to sqlite file, .sqlite included', required=True)
parser.add_argument('--chk', '--chunksize', type=int, help='chunk size', required=True)
parser.add_argument('--o', '--output_dir', type=str, help='output directory to save the npz file', required=True)

args = parser.parse_args()

# =================
# read arguments
# =================
sq_path = args.sq
chunk_s = args.chk
output_dir = args.o


chunk_size = chunk_s
f_path = sq_path


def readSITSData(chunk):
    """ 
    Read the data contained in name_file
        INPUT:
            - name_file: file where to read the data
        OUTPUT: 
        A npz file containing:
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - block_id: assigned random number for each block (splitted grid, 0-100 ) 
            - Y: label for eac sample
    """
    y_data = chunk.iloc[:,2]
    y = np.asarray(y_data.values, dtype='uint8')

    polygonID_data = chunk.iloc[:,3]
    polygon_ids = np.asarray(polygonID_data.values, dtype='uint16')

    block_id_data = chunk.iloc[:,4]
    block_ids = np.asarray(block_id_data.values, dtype='uint8')

    X_data = chunk.iloc[:,7:]
    X = np.asarray(X_data.values, dtype='uint16')

    return  X, polygon_ids, block_ids ,y


# Sqlite connection
conn = sqlite3.connect(f_path)

# read the data in chunks into variable list
X = [] # Explanatory variables
y = [] # Labels/codes
polygon_ids = [] # Polygon ids
block_ids = []

start_time = time.time()
# read the data in chunks
for chunk in pd.read_sql_query("select * from output;", conn,  chunksize=chunk_size):
    X_data, polygon_ids_data, block_ids_data, y_data = readSITSData(chunk)
    X.append(X_data)
    y.append(y_data)
    block_ids.append(block_ids_data)
    polygon_ids.append(polygon_ids_data)
print("read and append %s seconds ---" % (time.time() - start_time))
# convert lists to numpy arrays
start_time = time.time()
X = np.concatenate(X)
y = np.concatenate(y)
polygon_ids = np.concatenate(polygon_ids)
block_ids = np.concatenate(block_ids)
print("concatenate %s seconds ---" % (time.time() - start_time))

# get file sqlite name base to save the npz file; for example: "2018_sample_selection.sqlite" -> 2018
f = os.path.basename(f_path)
f_name = os.path.splitext(f)[0]
f_name = f_name.split('_')[0]

start_time = time.time()
# save X, y, polygon_ids in a single .npy file
np.savez_compressed(os.path.join(output_dir, '%s_SITS_data.npz' % f_name), X=X, y=y, block_id = block_ids, polygon_ids=polygon_ids)
print("npz compression %s seconds ---" % (time.time() - start_time))

# def load_npz(file_path):
#     """
#     Load data from a .npz file
#     """
#     with np.load(file_path) as data:
#         X = data['X']
#         y = data['y']
#         polygon_ids = data['polygon_ids']
#     return X, y, polygon_ids

#start_time = time.time()
#X, y, polygon_ids = load_npz(os.path.join(output_dir, '%s_SITS_data.npz' % f_name))
#print("Variables loading %s seconds ---" % (time.time() - start_time))
