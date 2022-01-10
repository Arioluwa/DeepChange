import numpy as np
import pandas as pd
import sqlite3
import time
import argparse
import os

# =================
# Command line arguments
# =================
parser = argparse.ArgumentParser(description='Read SITS data')

parser.add_argument('--sq', '--sqlitepath',type=str, help='path to sqlite file, .sqlite included', required=True)
parser.add_argument('--chk', '--chunksize', type=int, default=5000, help='chunk size')
parser.add_argument('--d', '--datepath', type=str, help='path to the gapfilled dates', required=True)
parser.add_argument('--o', '--output_dir', type=str, help='output directory to save the npz file', required=True)

args = parser.parse_args()

# =================
# read arguments
# =================
sq_path = args.sq
chunk_s = args.chk
output_dir = args.o
gfdate_path = args.d


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

# number of bands
n_channels = 10

# Sqlite connection
conn = sqlite3.connect(f_path)

# read the length of the data of time series
start_time = time.time()
L = len ((open(gfdate_path, 'r')).readlines())
print("getting L: %s seconds ---" % (time.time() - start_time))

# get the number of rows/pixels in the data
start_time = time.time()
N = pd.read_sql_query("select count(*) from output;", conn).values[0][0]
print("getting N: %s seconds ---" % (time.time() - start_time))

D = n_channels
# initialize the data arrays
X = np.zeros((N, D * L), dtype='uint16')
y = np.zeros((N), dtype='uint8')
polygon_ids = np.zeros((N), dtype='uint16')

start_time = time.time()
tmp = 0
for chunk in pd.read_sql_query("select * from output;", conn,  chunksize=chunk_size):
    # read the data in chunks into variable list
    X_data, polygon_ids_data, y_data = readSITSData(chunk)
    # concatenate the data
    X[tmp:tmp+len(X_data),:] = X_data
    y[tmp:tmp+len(y_data)] = y_data
    polygon_ids[tmp:tmp+len(polygon_ids_data)] = polygon_ids_data
    tmp += len(X_data)
print("reading variables: %s seconds ---" % (time.time() - start_time))


f = os.path.basename(f_path)
f_name = os.path.splitext(f)[0]
f_name = f_name.split('_')[0]

# save X, y, polygon_ids in a npz file
start_time = time.time()
np.savez_compressed(os.path.join(output_dir, '%s_SITS_data.npz' % f_name), X=X, y=y, polygon_ids=polygon_ids)
print("compressing time: %s seconds ---" % (time.time() - start_time))

def load_npz(file_path):
    """"
    Load npz data"""
    with np.load(file_path) as data:
        X = data['X']
        y = data['y']
        polygon_ids = data['polygon_ids']
    return X, y, polygon_ids

start_time = time.time()
X, y, polygon_ids = load_npz(os.path.join(output_dir, '%s_SITS_data.npz' % f_name))
print("loading time: %s seconds ---" % (time.time() - start_time))
