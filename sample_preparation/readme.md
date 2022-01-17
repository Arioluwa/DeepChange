# Content description

- Dataset description information
  - Number of samples per year
  - Class distribution per year (polygon-level)
  - Class distribution per year (pixels-level)
  - Dataset pixel-based confusion matrix between 2018 and 2019
- Grid split
- SQlite database sample extraction

The `readsits.py` script reads the sample extraction results from the SQLite database in chunks and returns the code/label (y), the polygon ID (polygon_id) and time series variable (X).

<!-- This folder contain `sample_stats.md` and `readsits.py` files.
The `sample_stats.md` reports the sample statistics, which includes;

- number of samples per year
- number of samples per class
- number of polygons per class
- Polygon intersection per class   -->
  <!-- Used the postgis `ST_Intersection` function to calculate the intersection of polygons per class, SQL script is available in `polygon_intersection.sql`. -->

<!-- https://gis.stackexchange.com/questions/339929/calculating-percentage-of-overlap-of-two-layers-in-qgis-3 -->

<!-- **Updated**

The `readsits.py` script reads the sample extraction results from the SQLite database in chunks and returns the code/label (y), the polygon ID (polygon_id) and time series variable (X).

- Time taken to read the sqlite database in chunks (chunk size = 50000): 3m 10.8s

`readsqlite.py` to be deleted.

~~The `readsqlite.py` script is used to reads the sample extraction results from the SQLite database in chunks, selecting the code/label (y), the polygon ID (polygon_id) and time series variable (x) using the `readSITSData()` function.~~

~~It further converts all the variables to numpy arrays and saved as a compressed .npz format. The variables are saved as .npz for fast loading and readability by other ML/DL frameworks. The choice of the compression format is based on the size of the data as it is expected to be large; which fails to read into memory while reading direclty from the sqlite db file. More information on the compression format can be found in the [here](https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk).~~

**Important observations:**
~~The script was tested on a subset sqlite database (5 images) with size of 2.7GB.~~

~~- Time taken to read the sqlite database in chunks (chunk size = 50000): 4m.11s~~

~~- Time taken to read the sqlite database in chunks (chunk size = 50000) and compress(npz format) and save file: 7m.58s~~

~~- Compression ratio: 3.4%~~
~~- Original size (sqlite): 2.7GB~~
~~- Compressed size (npz): 796MB~~

~~Loading the data (all the variables) into memory from the compressed .npz file: 10.8s~~ -->
