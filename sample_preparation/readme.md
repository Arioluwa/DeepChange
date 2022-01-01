# Content description

This folder contains two major files, `sample_stats.md` and `readsqlite.py`.  
The `sample_stats.md` reports the sample statistics, which includes;

- number of samples per year
- number of samples per class
- number of polygons per class
- Polygon intersection per class  
  Used the postgis `ST_Intersection` function to calculate the intersection of polygons per class, SQL script is available in `polygon_intersection.sql`.

<!-- https://gis.stackexchange.com/questions/339929/calculating-percentage-of-overlap-of-two-layers-in-qgis-3 -->

The `readsqlite.py` script is used to reads the sample extraction results from the SQLite database in chunks, selecting the code/label (y), the polygon ID (polygon_id) and time series variable (x) using the `readSITSData()` function.

It converts all the variables to numpy arrays and saves them in a compressed .npz format. The variables are saved as .npz for fast loading and readability by other ML/DL frameworks. The choice of the compression format is based on the size of the data as it is expected to be large, which fails to read into memory while reading directly from the SQLite db file. More information on the compression format can be found in the [here](https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk).

**Important observations:**  
The script was tested on a subset SQLite database (5 images) with a size of 2.7GB.

- Time taken to read the sqlite database in chunks (chunk size = 50000): 4m.11s

- Time taken to read the sqlite database in chunks (chunk size = 50000) and compress(npz format) and save file: 7m.58s

- Compression ratio: 3.4%
  - Original size (sqlite): 2.7GB
  - Compressed size (npz): 796MB

Loading the data (all the variables) into memory from the compressed .npz file: 10.8s
