#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import sqlite3
import os
import argparse

# =================
# Command line arguments
# =================
parser = argparse.ArgumentParser(description='Read SITS data')

parser.add_argument('--sq', '--sqlitepath',type=str, help='path to sqlite file, .sqlite included', required=True)
parser.add_argument('--chk', '--chunksize', type=int, default=500000, help='chunk size', required=True)
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
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - Y: label for each example
    """
    y_data = chunk.iloc[:,2]
    y = np.asarray(y_data.values, dtype='uint8')

    polygonID_data = chunk.iloc[:,4]
    polygon_ids = np.asarray(polygonID_data.values, dtype='uint16')

    X_data = chunk.iloc[:,6:]
    X = np.asarray(X_data.values, dtype='float32')

    return  X, polygon_ids, y


# Sqlite connection
conn = sqlite3.connect(f_path)

# read the data in chunks into variable list
X = [] # Explanatory variables
y = [] # Labels/codes
polygon_ids = [] # Polygon ids

# read the data in chunks
for chunk in pd.read_sql_query("select * from output;", conn,  chunksize=chunk_size):
    X_data, polygon_ids_data, y_data = readSITSData(chunk)
    X.append(X_data)
    y.append(y_data)
    polygon_ids.append(polygon_ids_data)

# convert lists to numpy arrays
X = np.concatenate(X)
y = np.concatenate(y)
polygon_ids = np.concatenate(polygon_ids)

# get file sqlite name base to save the npz file; for example: "2018_sample_selection.sqlite" -> 2018
f = os.path.basename(f_path)
f_name = os.path.splitext(f)[0]
f_name = f_name.split('_')[0]

# save X, y, polygon_ids in a single .npy file
np.savez_compressed(os.path.join(output_dir, '%s_SITS_data.npz' % f_name), X=X, y=y, polygon_ids=polygon_ids)