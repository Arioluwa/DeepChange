import numpy as np
import pandas as pd
import sqlite3


def readSITSData(file_path):
    """ 
    Read the data contained in name_file
        INPUT:
            - name_file: sqlite file where to read the data
        OUTPUT:
            - X: variable vectors for each example
            - polygon_ids: id polygon (use e.g. for validation set)
            - Y: label for each example
    """
    conn = sqlite3.connect(file_path)

    # read the data in chunks into variable list
    X = [] # Explanatory variables
    y = [] # Labels/codes
    polygon_ids = [] # Polygon ids


    # read the data in chunks
    for chunk in pd.read_sql_query("select * from output;", conn,  chunksize=50000):
        
        y_data = chunk.iloc[:,2]
        y_data = np.asarray(y_data.values, dtype='uint8')

        polygonID_data = chunk.iloc[:,4]
        polygonID_data = np.asarray(polygonID_data.values, dtype='uint16')

        X_data = chunk.iloc[:,6:]
        X_data = np.asarray(X_data.values, dtype='float32')  

        X.append(X_data)
        y.append(y_data)
        polygon_ids.append(polygonID_data)

    X = np.concatenate(X)
    y = np.concatenate(y)
    polygon_ids = np.concatenate(polygon_ids)
        
    return X, y, polygon_ids
        
