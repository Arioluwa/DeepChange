#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Summary

"""

import os
import os.path
import time
import re
import glob
from osgeo import ogr
import subprocess
import sys


def read_epsg(img_file):
	"""This function extracts EPSG code from image file.
		Using "gdalinfo" command to generate the gdal info text file.
		Read text file and return the last EPSG code from pattern AUTHORITY["EPSG","\d+"].
		Return the EPSG code in string format
	"""
	tmp_gdalinfo = os.path.join(os.path.split(img_file)[0], "tmp_gdalinfo.txt")
	cmd = "gdalinfo %s > %s" % (img_file,tmp_gdalinfo)
	if os.path.exists(tmp_gdalinfo):
		os.remove(tmp_gdalinfo)
	os.system(cmd)
	epsg_list = []
	#pattern = re.compile(r'AUTHORITY\[\"EPSG\",\"\d+\"\]') ###--- prev version of gdal
	pattern = re.compile(r'ID\[\"EPSG\",\d+\]')
	with open(tmp_gdalinfo) as input_f:
		for line in input_f.readlines():
			match = pattern.search(line)
			if match:
				epsg_list.append(match.group(0))
	try:
		epsg = re.search(r"\d+", epsg_list[-1]).group(0)
	except TypeError:
		print("No epsg code is available.")
	else:
		return str(epsg)




def reprojectionShp(shp_in, shp_out, crs_out="2154"):
	"""Reprojection shp_in in crs_out systeme, and save result in shp_out
        ARGs:
                - shp_in: input shapefile
                - shp_out: output shapefile
                - crs_out: output coordinate reference system
	"""
	if not os.path.exists(shp_out):
		cmd = 'ogr2ogr %s -t_srs EPSG:%s %s' % (shp_out, crs_out, shp_in)
		print(cmd)
		try:
			retcode = subprocess.check_call(cmd, shell=True)
		except subprocess.CalledProcessError as e:
			print("=================== Reprojection failed. ===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("====================================================================")
			sys.exit(1)
		else:
			print("============================== Done. ================================")
			print("====================================================================")




def clipVectorData(shp_in, shp_mask, shp_out, epsg_code='32754'):
	"""
        Cuts a shapefile with another shapefile
        ARGs:
                - shp_in: the shapefile to be cut
                - shp_mask: the other shapefile, mask
				- epsg_code: projection system
                - shp_out: output shapefile
        (From script_marcela/Decoupe.py)
	"""
	if os.path.exists(shp_out):
		os.remove(shp_out)
	cmd = "ogr2ogr -clipsrc %s %s %s -t_srs EPSG:%s -progress" % (shp_mask, shp_out, shp_in, epsg_code)
	print(cmd)
	try:
		retcode = subprocess.check_call(cmd, shell=True)
	except subprocess.CalledProcessError as e:
		print("=================== Cut ref data failed. ===================")
		print("Last command is:", e.cmd)
		print("Exit status: ", e.returncode)
		print("====================================================================")
		sys.exit(1)
	else:
		print("============================== Done. ================================")
		print("====================================================================")


def addField(shp_in, field_name, field_val):
	"""
		Add a field name with a unique value field_val [int] 
		ARGs:
			INPUT:
				- shp_in: input shapefile
	"""
	source = ogr.Open(shp_in, 1)
	layer = source.GetLayer()
	new_field = ogr.FieldDefn(field_name, ogr.OFTInteger)
	layer.CreateField(new_field)

	for i in layer:
		layer.SetFeature(i)
		i.SetField(field_name, field_val)
		layer.SetFeature(i)



def sample_stat_estimation(otb_path, img_file, shp_file, output_stat_xml, field):
	cmd = '%s/otbcli_PolygonClassStatistics -in %s -vec %s -out %s -field %s -ram 1000 -progress 1' % (
		otb_path, img_file, shp_file, output_stat_xml, field)
	if not os.path.exists(output_stat_xml):
		print(cmd)
		try:
			retcode = subprocess.check_call(cmd, shell=True)
		except subprocess.CalledProcessError as e:
			print("=================== Create stat.xml failed. ===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("====================================================================")
			sys.exit(1)
		else:
			print("============================== Done. ================================")
			print("====================================================================")
	else:
		print("%s has already existed." % output_stat_xml)




def sample_selection(otb_path, img_file, shp_file, stat_xml, output_sample_selection_sqlite, field):
	cmd = '%s/otbcli_SampleSelection -in %s -vec %s -instats %s -out %s -field %s -strategy all -ram 1000 -progress 1' % (
		otb_path, img_file, shp_file, stat_xml, output_sample_selection_sqlite, field)
	if not os.path.exists(output_sample_selection_sqlite):
		print(cmd)
		try:
			retcode = subprocess.check_call(cmd, shell=True)
		except subprocess.CalledProcessError as e:
			print("=================== Create sample.sqlite failed. ===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("====================================================================")
			sys.exit(1)
		else:
			print("============================== Done. ================================")
			print("====================================================================")
	else:
		print("%s has already existed." % output_sample_selection_sqlite)




def sample_extraction(otb_path, img_file, sample_selection_sqlite, output_sample_extract_sqlite, field, ram_processing=4000):
	cmd = '%s/otbcli_SampleExtraction -in %s -vec %s -out %s -outfield prefix -outfield.prefix.name band -field %s -ram %s -progress 1' % (
		otb_path, img_file, sample_selection_sqlite, output_sample_extract_sqlite, field.lower(), int(ram_processing/4))
	if not os.path.exists(output_sample_extract_sqlite):
		print(cmd)
		try:
			retcode = subprocess.check_call(cmd, shell=True)
		except subprocess.CalledProcessError as e:
			print("=================== Create extract.sqlite failed. ===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("====================================================================")
			sys.exit(1)
		else:
			print("============================== Done. ================================")
			print("====================================================================")
	else:
		print("%s has already existed." % output_sample_extract_sqlite)




def change_type_to_int(sqlite_file, bands_list):
	# Create columns string
	# column_string =
	column_string = "ogc_fid INTEGER, GEOMETRY BLOB, objectid INTEGER, lc_id INTEGER, lc2_id INTEGER, tileid INTEGER, originfid INTEGER, "
	for band in bands_list:
		column_string = column_string + band + " INTEGER, "
	column_string = column_string[:-2]
	sqlite_query = "sqlite3 %s 'CREATE TABLE newoutput (%s); INSERT INTO newoutput SELECT * FROM output; DROP TABLE output; ALTER TABLE newoutput RENAME TO output; VACUUM;' " % (sqlite_file, column_string)
	print(sqlite_query)
	try:
		retcode = subprocess.check_call(sqlite_query, shell=True)
	except subprocess.CalledProcessError as e:
		print("======================= Converting failed. =======================")
		print("Last command is:", e.cmd)
		print("Exit status: ", e.returncode)
		print("====================================================================")
		sys.exit(1)
	else:
		print("====================== Converting Finished ==========================")
		print("====================================================================")



def change_table_name(sqlite_file):
	sqlite_query = "sqlite3 %s 'ALTER TABLE newoutput RENAME TO output' " % sqlite_file
	try:
		retcode = subprocess.check_call(sqlite_query, shell=True)
	except subprocess.CalledProcessError as e:
		print("======================= Rename table failed. =======================")
		print("Last command is:", e.cmd)
		print("Exit status: ", e.returncode)
		print("====================================================================")
		sys.exit(1)
	else:
		print("======================= Rename table finished. =====================")
		print("====================================================================")
		print("\n")

def generate_sqlite_perdate(otb_path, perdate_path, shp_file, shp_mask, field, output_path, ram_processing=4000):
	"""Summary [need to be updated]:
		Generate sqlite file for each tile from the gapfilled image cube.
		Args:
			otb_path:
			img_file: path of the gapfilled image cube
			shp_file: path of the VLUIS shape file
			shp_mask: path of the shape file of interest area
			tile_lookup: path of the lookup talbe for tile ID and tile name
			field: Name of the field carrying the class name in the input vectors
			output_path: the output path of sqlite file
	:return:
	"""
	perdate_original_path = os.path.join(perdate_path, "Original")
	if not os.path.exists(perdate_original_path):
		os.mkdir(perdate_original_path)
	sqlite_perdate_original_path = os.path.join(perdate_original_path, "sqlite")
	if not os.path.exists(sqlite_perdate_original_path):
		os.mkdir(sqlite_perdate_original_path)
	list_perdate_img_file = glob.glob(perdate_original_path + "/*.tif")
	print("list_perdate_img_file: ", list_perdate_img_file) #TODO: dlt
	
	tile_name = os.path.split(list_perdate_img_file[0])[1][0:6]
	tile_output_path = output_path
	tile_output_tmp_path = os.path.join(tile_output_path, "tmp")
	if not os.path.exists(tile_output_tmp_path):
		os.mkdir(tile_output_tmp_path)
	print("tile_name: ", tile_name) #TODO: dlt
	
	# Read EPSG (projection system) from gapfilled image
	print("====================================================================")
	print("========================= Read EPSG ================================")
	epsg = read_epsg(list_perdate_img_file[0])
	print("The projection system code is %s." %epsg)
	print("====================================================================")


	# Reproject ref data (actually reproject ref data doesn't need to process every time.)
	print("====================================================================")
	print("======================= Reproject ref data ==========================")
	reproject_shp_file = os.path.join(tile_output_tmp_path, "reproject_" + os.path.basename(shp_file))
	reprojectionShp(shp_file, reproject_shp_file, epsg)
	print("\n")
	print("======================= Reproject mask data ==========================")
	reproject_mask_file = os.path.join(tile_output_tmp_path, "reproject_" + os.path.basename(shp_mask))
	reprojectionShp(shp_mask, reproject_mask_file, epsg)
	
	# Cut ref data according to the extent shape file(_PRIO.shp)
	print("====================================================================")
	print("========================= Cut ref data ==============================")
	cut_reproject_shp_file = os.path.join(tile_output_tmp_path, "cut_" + os.path.basename(reproject_shp_file))
	if not os.path.exists(cut_reproject_shp_file):
		clipVectorData(reproject_shp_file, reproject_mask_file, cut_reproject_shp_file, epsg)
	
	
	# Create polygon_stat.xml
	print("====================================================================")
	print("===================== Create polygon_stat.xml ======================")
	output_stat_xml = os.path.join(tile_output_tmp_path, tile_name + "_polygon_classes_stats.xml")
	if not os.path.exists(output_stat_xml):
		sample_stat_estimation(otb_path, list_perdate_img_file[0], cut_reproject_shp_file, output_stat_xml, field)
	
	# Create sample.sqlite
	print("====================================================================")
	print("====================== Create sample.sqlite =========================")
	output_sample_selection_sqlite = os.path.join(tile_output_tmp_path, tile_name + "_sample_selection.sqlite")
	if not os.path.exists(output_sample_selection_sqlite):
		sample_selection(otb_path, list_perdate_img_file[0], cut_reproject_shp_file, output_stat_xml, output_sample_selection_sqlite, field)
	
	# Create 1 extract.sqlite per date
	# Create extract.sqlite
	print("====================================================================")
	print("===================== Create extract.sqlite ========================")
	for add, img_file in enumerate(list_perdate_img_file):
		#print("add: ", add)
		#print("img_file: ", img_file)
		img_file_base = os.path.basename(img_file)
		split_im_file = img_file_base.split('_')
		#print("split_im_file: ", split_im_file)
		dd = "0"
		for el in split_im_file:
			#print("el: ", el)
			if len(el)==8 and el[:2]=="20":
				dd = el
				break
			#print("dd: ", dd)
		if dd == "0": ##-- not an image file
			continue
		#print(1/0)
		output_sample_extract_sqlite = os.path.join(sqlite_perdate_original_path, tile_name + "_sample_extract_" + dd + ".sqlite")
		if not os.path.exists(output_sample_extract_sqlite):
			sample_extraction(otb_path, img_file, output_sample_selection_sqlite, output_sample_extract_sqlite, field, ram_processing)
	
	# Cleaning the sqlite files and convert them to CSV
	#---- to Jupyter notebook
	
	
	#remove temporary files
	if 0:
		try:
			retcode = subprocess.check_call(["rm", '-r', tile_output_tmp_path])
		except subprocess.CalledProcessError as e:
			print("================== Remove temporary files failed.===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("===================================================================================")
			sys.exit(1)
		else:
			print("=================== Remove temporary files done. ==========================")
			print("===================================================================================")

	
	

def generate_sqlite(otb_path, img_file, shp_file, shp_mask, tile_lookup, field, output_path):
	"""Summary:
		Generate sqlite file for each tile from the gapfilled image cube.
		Args:
			otb_path:
			img_file: path of the gapfilled image cube
			shp_file: path of the VLUIS shape file
			shp_mask: path of the shape file of interest area
			tile_lookup: path of the lookup talbe for tile ID and tile name
			field: Name of the field carrying the class name in the input vectors
			output_path: the output path of sqlite file
	:return:
	"""
	tile_name = os.path.split(img_file)[1][0:6]
	tile_output_path = output_path
	tile_output_tmp_path = os.path.join(tile_output_path, "tmp")
	if not os.path.exists(tile_output_tmp_path):
		os.mkdir(tile_output_tmp_path)

	# Read EPSG (projection system) from gapfilled image
	print("====================================================================")
	print("========================= Read EPSG ================================")
	epsg = read_epsg(img_file)
	print("The projection system code is %s." %epsg)
	print("====================================================================")


	# Reproject ref data (actually reproject ref data doesn't need to process every time.)
	print("====================================================================")
	print("======================= Reproject ref data ==========================")
	reproject_shp_file = os.path.join(tile_output_tmp_path, "reproject_" + os.path.basename(shp_file))
	reprojectionShp(shp_file, reproject_shp_file, epsg)
	print("\n")
	print("======================= Reproject mask data ==========================")
	reproject_mask_file = os.path.join(tile_output_tmp_path, "reproject_" + os.path.basename(shp_mask))
	reprojectionShp(shp_mask, reproject_mask_file, epsg)
	


	# Cut ref data according to the extent shape file(_PRIO.shp)
	print("====================================================================")
	print("========================= Cut ref data ==============================")
	cut_reproject_shp_file = os.path.join(tile_output_tmp_path, "cut_" + os.path.basename(reproject_shp_file))
	clipVectorData(reproject_shp_file, reproject_mask_file, cut_reproject_shp_file, epsg)
	


	# Add tile_id attribute to shapefile.
	"""
	Unique ID for Brittany dataset
	print("====================================================================")
	print("===================== Add tile_id attribute =========================")
	with open(tile_lookup, "r") as input_f:
		for line in input_f.readlines():
			clean_line = line.strip()
			if re.search(r"\d{2}[A-Z]{3}", clean_line):
				if re.search(r"\d{2}[A-Z]{3}", clean_line).group(0) == tile_name:
					field_val = int(re.search(r"^\d+",line).group(0))
	addField(cut_reproject_shp_file, "tileID", field_val)
	"""
	

	# Create polygon_stat.xml
	print("====================================================================")
	print("===================== Create polygon_stat.xml ======================")
	output_stat_xml = os.path.join(tile_output_tmp_path, tile_name + "_polygon_classes_stats.xml")
	sample_stat_estimation(otb_path, img_file, cut_reproject_shp_file, output_stat_xml, field)
	

	# Create sample.sqlite
	print("====================================================================")
	print("====================== Create sample.sqlite =========================")
	output_sample_selection_sqlite = os.path.join(tile_output_tmp_path, tile_name + "_sample_selection.sqlite")
	sample_selection(otb_path, img_file, cut_reproject_shp_file, output_stat_xml, output_sample_selection_sqlite, field)
	
	

	# Create extract.sqlite
	print("====================================================================")
	print("===================== Create extract.sqlite ========================")
	output_sample_extract_sqlite = os.path.join(tile_output_path, tile_name + "_sample_extract.sqlite")
	sample_extraction(otb_path, img_file, output_sample_selection_sqlite, output_sample_extract_sqlite, field)
	

	# Change the type of data values in extract.sqlite from float to int
	"""
	print("====================================================================")
	print("======================= Convert data type ===========================")
	bands_list = ["band%d" % i for i in range(730)]
	change_type_to_int(output_sample_extract_sqlite, bands_list)
	
	
	# Due to requirement of otbcli_TrainImageClassifier, Change table name to output 
	print("====================================================================")
	print("===================== Change table name ========================")
	change_table_name(output_sample_extract_sqlite)
	"""
	
	#remove temporary files
	if 0:
		try:
			retcode = subprocess.check_call(["rm", '-r', tile_output_tmp_path])
		except subprocess.CalledProcessError as e:
			print("================== Remove temporary files failed.===================")
			print("Last command is:", e.cmd)
			print("Exit status: ", e.returncode)
			print("===================================================================================")
			sys.exit(1)
		else:
			print("=================== Remove temporary files done. ==========================")
			print("===================================================================================")

if __name__ == "__main__":
	"""
	Test Zehui:
	otb_path = "/home/cacao/iota2/scripts/install/OTB/build/OTB/build/bin"
	img_file = "/home/zehui/summerproject/data/SeagateBlack/SeagateBlack/zehui/test/54HWC/54HWC_GapFilled_Image.tif"
	shp_file = "/home/zehui/summerproject/data/SeagateBlack/SeagateBlack/ref_data_with_urban_polygon/final_ref_data/VLUIS_GroundData_2017.shp"
	shp_mask = '/home/zehui/summerproject/data/T54HXE/PEPS_MAJA/ref_data/tiles/54HWC_mask_tile_PRIO.shp'
	output_path = "/home/zehui/summerproject/data/SeagateBlack/SeagateBlack/zehui/test/54HWC"
	tile_lookup = "/home/zehui/summerproject/data/SeagateBlack/SeagateBlack/tiles_lookuptable.csv"
	field = "lc_id"
	generate_sqlite(otb_path, img_file, shp_file, shp_mask, tile_lookup, field, output_path)
	"""
	#-- Test Charlotte
	otb_path = "/home/cpelleti/OTB/OTB-7.0.0-Linux64/bin"
	img_file = "/data/BreizhCrops/ExpeROI/data/T30UVU_Image_ROI_ox4000_oy3000_sx1000_sy1000.tif"
	shp_file = "/data/BreizhCrops/ExpeROI/test_code/crops.shp"
	shp_mask = '/data/BreizhCrops/ExpeROI/test_code/roi_extent_2154.shp'
	output_path = "/data/BreizhCrops/ExpeROI/test_code/output/"
	tile_lookup = None
	field = "CODE_CULTU"
	print("Testing generate_sqlite.py")
	generate_sqlite(otb_path, img_file, shp_file, shp_mask, tile_lookup, field, output_path)
	
	
#EOF	
