## Folder description and contents  
---
Found two approcahes to download SENTINAL 2 L2A data from Theia:
- [theia_download](https://github.com/olivierhagolle/theia_download) source: https://theia.cnes.fr/atdistrib/rocket/#/help
    - command: python3 theia_download.py -c SENTINEL2 -t T31TCJ -a config_theia.cfg -r 51 --level 80 LEVEL2A -d 2018-01-01 -f 2018-02-27
    - Limitation: Token only valid for two hour.
- [EODAG](https://github.com/CS-SI/eodag)
    - edit script.py to the start date and end date, and cloud cover
    - auxdata is needed to filter based on tile_id
    - Works fine so far.

```
├── data_download
│   ├── eodag
│   │   ├── auxdata
│   │   │   └── sentinel2_tiling_grid_centroids.zip
│   │   └── script.py
│   ├── read.md
│   └── theia_download
│       ├── README.md
│       ├── config_landsat.cfg
│       ├── config_theia.cfg
│       └── theia_download.py
```

## File Sentinel 2 Level2A Download
---
Tile - T31TCJ  
RelativeOrbitNumber - 51  
Coverage - Full   
- 2018
    - 37

- 2019
    - 33

Total 70
