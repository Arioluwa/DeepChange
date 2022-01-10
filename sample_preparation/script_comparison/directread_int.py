# Initialization matrix to the cluster
import sqlite3
import numpy as np
import pandas as pd
import time

n_channels = 10

def readSITSData(sqfile_path, gfdate_path):

    start_time = time.time()
    L = len((open(gfdate_path, 'r').readlines()))
    print("get L: %s seconds ---" % (time.time() - start_time))
    
    conn = sqlite3.connect(sqfile_path)
    start_time = time.time()
    # get the total number of rows/ pixels
    N  = pd.read_sql_query("select COUNT(*) from output;" , conn).values[0][0]
    print("get the number of rows: %s seconds ---" % (time.time() - start_time))

    D = n_channels
    # initialize the matrix shape
    X = np.zeros((N, L * D), dtype='uint16')
    y = np.zeros((N), dtype='uint8')
    polygon_ids = np.zeros((N), dtype='uint16')


    # read the data in chunks into variable list
    start_time = time.time()
    tmp = 0
    for chunk in pd.read_sql_query("select * from output;", conn,  chunksize=5000): # countrs thru index in 5000 steps
        
        y_data = chunk.iloc[:,2]
        y_data = np.asarray(y_data.values, dtype='uint8')

        polygonID_data = chunk.iloc[:,4]
        polygonID_data = np.asarray(polygonID_data.values, dtype='uint16')

        X_data = chunk.iloc[:,6:]
        X_data = np.asarray(X_data.values, dtype='uint16')
        
        X[tmp:tmp+len(X_data),:] = X_data
        y[tmp:tmp+len(y_data)] = y_data
        polygon_ids[tmp:tmp+len(polygonID_data)] = polygonID_data

        # steps 
        tmp = tmp + X_data.shape[0]
    print("read all variables %s seconds ---" % (time.time() - start_time))
    return X, y, polygon_ids
    
