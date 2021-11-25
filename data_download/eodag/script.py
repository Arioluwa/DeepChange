#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
###############################################################################
import os
from zipfile import ZipFile
from eodag import EODataAccessGateway
import shapefile

###############################################################################

default_search_criteria = dict(
    productType="S2_MSI_L2A_MAJA",
    start="2018-01-01",
    end="2019-12-31",
)

workspace = "eodag_workspace_locations_tiles"
if not os.path.isdir(workspace):
    os.mkdir(workspace)

sentinel2_grid_zip = os.path.join("auxdata", "sentinel2_tiling_grid_centroids.zip")
if not os.path.isfile(sentinel2_grid_zip):
    raise FileNotFoundError("Auxdata not found, please check your configuration.")


with ZipFile(sentinel2_grid_zip, "r") as fzip:
    fzip.extractall("auxdata")


sentinel2_shp = os.path.join("auxdata", "sentinel2_tiling_grid_centroids.shp")
with shapefile.Reader(sentinel2_shp) as shp:
    shaperecs = shp.shapeRecords()


# Save the locations configuration file.
locations_yaml_content = """
shapefiles:
  - name: s2_tile_centroid
    path: {}
    attr: tile_id
""".format(
    os.path.abspath(sentinel2_shp)
)

locations_filepath = os.path.abspath(os.path.join(workspace, "custom_locations.yml"))

with open(locations_filepath, "w") as f_yml:
    f_yml.write(locations_yaml_content.strip())

dag = EODataAccessGateway(locations_conf_path=locations_filepath)
dag.set_preferred_provider("theia")

products = dag.search_all(
    locations=dict(s2_tile_centroid="31TCJ"),
    relativeOrbitNumber=51,
    **default_search_criteria
)

dag.download_all(products)

__name__ == "__main__" and dag.download_all(products)
