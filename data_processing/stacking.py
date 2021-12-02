#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Summary:
	This script processes the original L2A products for each tile to one single 
	gapfilled image, and provides three kinds of reviews of original images, masks
	and gapfilled images.
	
	The process contains a series of data manipulation and transformation. The detail 
	please find at https://.......

"""

import os
import os.path
import sys

# import optparse
# import time
import re
import glob
import subprocess


# Read files
# 1: read original images
def sort_original_image_folders(input_path):
    """Read L2A images folders
    return a list of (date, image folder path) tuples sorted by date
    """
    image_files_list = os.listdir(input_path)
    list_of_tuples = []
    date_pattern = re.compile(r"\d{8}")
    for image_file in image_files_list:
        image_file_date = date_pattern.search(image_file).group(0)
        image_file_path = os.path.join(input_path, image_file)
        list_of_tuples.append((image_file_date, image_file_path))
    list_of_tuples.sort()  # sort by date
    # print to show sorted images
    print("======================================================================")
    print("\n")
    print("Sorted image files by dates:")
    print("\n")
    for element in list_of_tuples:
        print("Date: %s, Abs path: %s" % (element[0], element[1]))
    print("======================================================================")
    print("\n")
    return list_of_tuples
    # can modify to return a full list of images


def only_select_tif_file(a_file):
    """If a file is a tif file, return true,
    otherwise, return false
    """
    return os.path.splitext(a_file)[1] == ".tif"


def only_select_target_bands(a_file):
    """If a file is within the target bands, return true,
         target_bands_list = 'FRE_B2', 'FRE_B3', 'FRE_B4', 'FRE_B5', 'FRE_B6', 'FRE_B7', 'FRE_B8', 'FRE_B8A', 'FRE_B11', 'FRE_B12'
    otherwise, return false.
    """
    if not re.search(r"FRE_B([2-8]|8A|11|12).tif$", a_file):
        return False
    else:
        return True


def read_original_image_folders(list_of_tuple, tmp_path):
    """Given a list of sorted (date, image folder path) tuples,
    Go into each folder and select target tif files (10 bands per acquisition date),
    Order of bands: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    Return a full list of sorted target tif file paths
    """

    full_images_list = []
    print("======================================================================")
    print("===================Print paths of input images========================")

    for date_image_tuple in list_of_tuple:

        image_folder_path = date_image_tuple[1]
        cur_date = date_image_tuple[0]
        list_of_any_files = os.listdir(image_folder_path)
        # -- Search in unzip files (one folder per date acquisition)
        list_of_select_files = []
        for a_file in list_of_any_files:
            a_file_path = os.path.join(image_folder_path, a_file)
            if os.path.isfile(a_file_path) and only_select_target_bands(a_file_path):
                list_of_select_files.append(a_file_path)
        # -- Search in tmp files (all files together => additional condition on current dates)
        list_of_any_files = os.listdir(tmp_path)
        for a_file in list_of_any_files:
            a_file_path = os.path.join(tmp_path, a_file)
            if (
                os.path.isfile(a_file_path)
                and only_select_target_bands(a_file_path)
                and only_select_cur_date(a_file_path, cur_date)
            ):
                list_of_select_files.append(a_file_path)
        # -- Sort apply on band number as 20m bands can be present twice
        bands = ["2", "3", "4", "5", "6", "7", "8", "8A", "11", "12"]
        list_of_ordered_select_files = []
        for bb in bands:
            res = [i for i in list_of_select_files if "B" + bb + "." in i]
            print(res)
            if len(res) == 1:
                list_of_ordered_select_files.append(res[0])
            else:  # 20 m band + resampled version (select the resampled version)
                list_of_ordered_select_files.append([i for i in res if "tmp/" in i][0])
        for afile in list_of_ordered_select_files:
            full_images_list.append(afile)
    for f in full_images_list:
        print(f)
    print("======================================================================")
    return full_images_list


# 2: read original masks
def read_original_masks(image_folder_path):
    """Receive single image file path
    Return a list of abs path of CLM or EDG or SAT.
    """
    mask_file_path = image_folder_path + "/MASKS"
    mask_list = [i for i in os.listdir(mask_file_path) if only_select_tif_file(i)]
    selected_masks_path_list = []
    clm_pattern = re.compile(r"CLM_R1")
    edf_pattern = re.compile(r"EDG_R1")
    sat_pattern = re.compile(r"SAT_R1")
    for mask in mask_list:
        if clm_pattern.search(mask):
            selected_masks_path_list.append(os.path.join(mask_file_path, mask))
        elif edf_pattern.search(mask):
            selected_masks_path_list.append(os.path.join(mask_file_path, mask))
        elif sat_pattern.search(mask):
            selected_masks_path_list.append(os.path.join(mask_file_path, mask))
    selected_masks_path_list.sort()
    return selected_masks_path_list


# 3: write original dates
def write_original_dates(date_image_tuple, output_path):
    original_dates_file = os.path.join(output_path, "original_dates.txt")
    if not os.path.exists(original_dates_file):
        print("=============================================================")
        print("\n")
        print(
            "Write original acquisition dates of images into %s" % original_dates_file
        )
        with open(original_dates_file, "w") as f_output:
            for element in date_image_tuple:
                print(element[0])
                f_output.write(element[0] + "\n")
        f_output.close()
        print("The original dates are written successfully.")
        print("=============================================================")
        print("\n")
    return original_dates_file


def read_dates(date_file):
    """
    Read dates from date_file
    Return a list of dates (string format)
    """
    with open(date_file) as f:
        content = f.readlines()
    return [d.strip() for d in content]


# Convert to binary mask
def convert_mask_to_binary(otb_path, mask_path_list, binary_mask_path, ram_processing):
    """Convert CLM, EDG and SAT masks to binary mask using OTB command
            0: Valid pixel
            1: Invalid pixel
    And concatenate them into one tif image.
    """
    # command = "%s/otbcli_BandMathX -il %s -out %s uint8 -exp '(im1b1>0?1:0);(im2b1==1?1:0);(im3b1==1?1:0)' -ram 1000 -progress 1" % (otb_path, " ".join(mask_path_list), binary_mask_path)
    command = (
        "%s/otbcli_BandMath -il %s -out %s uint8 -exp '(im1b1>0?1:0) || (im2b1==1?1:0) || (im3b1==1?1:0)' -ram %s -progress 1"
        % (otb_path, " ".join(mask_path_list), binary_mask_path, ram_processing)
    )
    print(command)
    if not os.path.exists(binary_mask_path):
        try:
            retcode = subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Mask converting failed =======================")
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "===================================================================="
            )
            sys.exit(1)
        else:
            # # New feature: delete original masks after convert
            # try:
            # rm_list_str = " ".join(mask_path_list)
            # cmd = "rm -f %s" % rm_list_str
            # retcode = subprocess.check_call(cmd, shell=True)
            # except subprocess.CalledProcessError as e:
            # print("================= Deleting original masks failed ====================")
            # print("Last command is:", e.cmd)
            # print("Exit status: ", e.returncode)
            # print("====================================================================")
            # sys.exit(1)
            # else:
            print(
                "===================== Mask converting done. ========================"
            )
            print(
                "===================================================================="
            )
    else:
        print("%s has already existed." % binary_mask_path)


# Resample 20m to 10m
# before this step, only tif files should be selected.
# assume paths are abs path.
def only_select_20_band(image_path):
    """receive an image_path,
    return true if the resolution of this image is 20m,
    otherwise, return false.
    """
    #  r20_bands_list: ['FRE_B5', 'FRE_B6', 'FRE_B7', 'FRE_B8A', 'FRE_B11', 'FRE_B12']
    if not re.search(r"FRE_B([5-7]|11|12|8A).tif$", image_path):
        return False
    else:
        return True


def only_select_cur_date(image_path, cur_date):
    """receive an image_path and cur_date,
    return true if the image_path has been acquired at cur_date,
    otherwise, return false.
    """
    my_regex = re.escape(cur_date) + r".*.tif$"
    if not re.search(my_regex, image_path):
        return False
    else:
        return True


def resample_20_to_10(otb_path, full_image_path_list, tmp_path, ram_processing):
    """Resample 20m to 10m using OTB
    Receive full image_path_list
    Only resample r20m images
    Return new image path list with all in 10m resolution
    """
    r20_image_path_list = [i for i in full_image_path_list if only_select_20_band(i)]
    r10_image_path_list = []
    for image in r20_image_path_list:
        new_r10_path = os.path.join(tmp_path, os.path.basename(image))
        ref_image = re.sub(r"FRE_B\d{1,2}A?\.tif$", "FRE_B2.tif", image)
        command = (
            "%s/otbcli_Superimpose -inr %s -inm %s -out %s uint16 -interpolator nn -ram %s -progress 1"
            % (otb_path, ref_image, image, new_r10_path, ram_processing)
        )
        if not os.path.exists(new_r10_path):
            print(command)
            try:
                retcode = subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError as e:
                print("==================== Resampling failed =======================")
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "===================================================================="
                )
                sys.exit(1)
            else:
                print(
                    "======================= Resampling done. ============================"
                )
                print(
                    "===================================================================="
                )
                # ~ # New feature: delete original r20 image after resampling
                # ~ try:
                # ~ cmd = "rm -f %s" % image
                # ~ retcode = subprocess.check_call(cmd, shell=True)
                # ~ except subprocess.CalledProcessError as e:
                # ~ print("================ Deleting original r20 image failed ================")
                # ~ print("Last command is:", e.cmd)
                # ~ print("Exit status: ", e.returncode)
                # ~ print("====================================================================")
                # ~ sys.exit(1)
                # ~ else:
                # ~ print("======================= Resampling done. ============================")
                # ~ print("====================================================================")
        else:
            print("%s has already existed." % new_r10_path)
        r10_image_path_list.append(new_r10_path)

    # replace r20 image path with new r10 image path in full image path list
    for i in range(len(full_image_path_list)):
        for j in range(len(r20_image_path_list)):
            if full_image_path_list[i] == r20_image_path_list[j]:
                full_image_path_list[i] = r10_image_path_list[j]
                break
    return full_image_path_list


# Method : first stack images (or masks) into single tif, then gapfill it
# 1： Stack images according to dates
def stack_images_to_tif(otb_path, image_list, image_tif_path, ram_processing):
    if len(image_list) < 801:
        command = (
            "%s/otbcli_ConcatenateImages -il %s -out %s uint16 -ram %s -progress 1"
            % (otb_path, " ".join(image_list), image_tif_path, ram_processing)
        )
        if not os.path.exists(image_tif_path):
            # print(command)
            try:
                retcode = subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError as e:
                print(
                    "==================== Image stacking failed ======================="
                )
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "===================================================================="
                )
                sys.exit(1)
            else:
                print(
                    "=================  Image stacking done. ============================"
                )
                print(
                    "===================================================================="
                )
                # ~ rm_image_list = " ".join(image_list)
                # ~ cmd = "rm -f %s" % rm_image_list
                # ~ try:
                # ~ retcode = subprocess.check_call(cmd, shell=True)
                # ~ except subprocess.CalledProcessError as e:
                # ~ print("==================== Removing unstacked images failed =======================")
                # ~ print("Last command is:", e.cmd)
                # ~ print("Exit status: ", e.returncode)
                # ~ print("====================================================================")
                # ~ sys.exit(1)
                # ~ else:
                # ~ print("=================  Image stacking done. ============================")
                # ~ print("====================================================================")
        else:
            # print("The stacked image of tile %s has already existed at %s ." % (image_tif_path.strip('/')[-2], image_tif_path))
            print("The stacked image has already existed at %s ." % image_tif_path)
    else:
        image_list1 = image_list[:800]
        image_list2 = image_list[800:]
        # create temp files
        image_tif_path1 = os.path.join(
            os.path.dirname(image_tif_path), "t1_" + os.path.basename(image_tif_path)
        )
        image_tif_path2 = os.path.join(
            os.path.dirname(image_tif_path), "t2_" + os.path.basename(image_tif_path)
        )
        # stack two half one by one.
        stack_images_to_tif(otb_path, image_list1, image_tif_path1, ram_processing)
        stack_images_to_tif(otb_path, image_list2, image_tif_path2, ram_processing)
        # stack two half together
        new_image_list = [image_tif_path1, image_tif_path2]
        stack_images_to_tif(otb_path, new_image_list, image_tif_path, ram_processing)

        # delete temp files
        os.system("rm %s" % image_tif_path1)
        os.system("rm %s" % image_tif_path2)


# 1bis [Charlotte]： Stack images per data
def stack_images_per_date_to_tif(
    otb_path,
    image_list,
    tile,
    image_tif_path,
    list_of_tuple,
    masks_list_for_summask,
    mask_flag=False,
    ram_processing=4000,
):
    nbands = 10  # TODO global variable or band definition
    print(image_list)
    print("##########")
    for add, date_image_tuple in enumerate(list_of_tuple):
        date_image_tif_path = os.path.join(
            image_tif_path, "%s_%s_Image.tif" % (tile, date_image_tuple[0])
        )
        if mask_flag:
            command = (
                "%s/otbcli_ConcatenateImages -il %s -out %s uint16 -ram %s -progress 1"
                % (
                    otb_path,
                    " ".join(image_list[add * nbands : (add + 1) * nbands])
                    + " "
                    + masks_list_for_summask[add],
                    date_image_tif_path,
                    ram_processing,
                )
            )
        else:
            command = (
                "%s/otbcli_ConcatenateImages -il %s -out %s uint16 -ram %s -progress 1"
                % (
                    otb_path,
                    " ".join(image_list[add * nbands : (add + 1) * nbands]),
                    date_image_tif_path,
                    ram_processing,
                )
            )
        if not os.path.exists(date_image_tif_path):

            print(command)
            try:
                retcode = subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError as e:
                print(
                    "==================== Image stacking failed ======================="
                )
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "===================================================================="
                )
                sys.exit(1)
            else:
                print(
                    "=================  Image stacking done. ============================"
                )
                print(
                    "===================================================================="
                )
                # ~ rm_image_list = " ".join(image_list)
                # ~ cmd = "rm -f %s" % rm_image_list
                # ~ try:
                # ~ retcode = subprocess.check_call(cmd, shell=True)
                # ~ except subprocess.CalledProcessError as e:
                # ~ print("==================== Removing unstacked images failed =======================")
                # ~ print("Last command is:", e.cmd)
                # ~ print("Exit status: ", e.returncode)
                # ~ print("====================================================================")
                # ~ sys.exit(1)
                # ~ else:
                # ~ print("=================  Image stacking done. ============================")
                # ~ print("====================================================================")
        else:
            # print("The stacked image of tile %s has already existed at %s ." % (image_tif_path.strip('/')[-2], image_tif_path))
            print("The stacked image has already existed at %s ." % image_tif_path)
    # ~ else:
    # ~ image_list1 = image_list[:800]
    # ~ image_list2 = image_list[800:]
    # ~ # create temp files
    # ~ image_tif_path1 = os.path.join(os.path.dirname(image_tif_path), "t1_" + os.path.basename(image_tif_path))
    # ~ image_tif_path2 = os.path.join(os.path.dirname(image_tif_path), "t2_" + os.path.basename(image_tif_path))
    # ~ # stack two half one by one.
    # ~ stack_images_to_tif(otb_path, image_list1, image_tif_path1, ram_processing)
    # ~ stack_images_to_tif(otb_path, image_list2, image_tif_path2, ram_processing)
    # ~ # stack two half together
    # ~ new_image_list = [image_tif_path1, image_tif_path2]
    # ~ stack_images_to_tif(otb_path, new_image_list, image_tif_path, ram_processing)

    # ~ # delete temp files
    # ~ os.system("rm %s" % image_tif_path1)
    # ~ os.system("rm %s" % image_tif_path2)


# 2： Stack masks
def stack_masks_to_tif(otb_path, mask_list, mask_tif_path, ram_processing):
    command = "%s/otbcli_ConcatenateImages -il %s -out %s uint8 -ram %s -progress 1" % (
        otb_path,
        " ".join(mask_list),
        mask_tif_path,
        ram_processing,
    )
    if not os.path.exists(mask_tif_path):
        # print(command)
        try:
            retcode = subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Mask stacking failed =======================")
            print(
                "===================================================================="
            )
            sys.exit(1)
        else:
            print("=================  Mask stacking done. ============================")
            print(
                "===================================================================="
            )
    else:
        # print("The stacked mask of Tile %s has already existed at %s ." % (mask_tif_path.strip('/')[-2], mask_tif_path))
        print("The stacked mask has already existed at %s ." % mask_tif_path)


# 3： Gapfilling
def image_timeseries_gapfilling(
    otb_path,
    image_tif,
    mask_tif,
    output_tif,
    original_dates_file,
    out_dates_list_file,
    ram_processing,
):
    """Perform gapfilling
    -in: input image
    -mask: input mask
    -out: output gapfilled image
    -comp: compPerDate
    -it: Gapfilling method
    -id: DateListI
    -od: DateListO
    """

    command = (
        "%s/otbcli_ImageTimeSeriesGapFilling -in %s -mask %s -out %s uint16 -comp 10 -it linear -id %s -od %s -ram %s -progress 1"
        % (
            otb_path,
            image_tif,
            mask_tif,
            output_tif,
            original_dates_file,
            out_dates_list_file,
            ram_processing,
        )
    )
    if not os.path.exists(output_tif):
        print(command)
        try:
            retcode = subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Gapfilling failed =======================")
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "===================================================================="
            )
            sys.exit(1)
        else:
            print("=================  Gapfilling done. ============================")
            print(
                "===================================================================="
            )
    else:
        print(
            "The gapfilled image of Tile %s has already existed at %s ."
            % (output_tif.strip("/")[-2], output_tif)
        )


# 4: Preview
def only_select_3_bands(a_file):
    """If a file is within the target bands, return true,
        arget_bands_list = ['FRE_B3', 'FRE_B4', 'FRE_B8']
    otherwise, return false.
    """
    if not re.search(r"FRE_B[348].tif$", a_file):
        return False
    return True


def rescale_to_1000_1000(otb_path, input_image, output_image, ram_preview):
    """This function receives a image,
    rescale it into 1000*1000,
    output the scaled image.
    """
    cmd_rescaling = (
        "%s/otbcli_RigidTransformResample -in %s -out %s -transform.type.id.scalex 0.1 -transform.type.id.scaley 0.1 uint16 -ram %s -progress 1"
        % (otb_path, input_image, output_image, ram_preview)
    )
    if not os.path.exists(output_image):
        print(cmd_rescaling)
        try:
            retcode = subprocess.check_call(cmd_rescaling, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Rescaling failed =======================")
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "===================================================================="
            )
            sys.exit(1)
        else:
            print(
                "======================  Rescaling done. ============================"
            )
            print(
                "===================================================================="
            )
    else:
        print("%s has already been scaled." % output_image)


def convert_to_jpg(otb_path, input_image_tif, output_image_jpg, ram_preview):
    """This function receives a tif image,
    convert it into false color image in jpg format,
    in jpg, the red is band8(NIR), the green is band4(red), the blue is band3(green).
    """
    cmd_convert_to_jpg = (
        "%s/otbcli_DynamicConvert -in %s -out %s -quantile -channels rgb -channels.rgb.red 3 -channels.rgb.green 2 -channels.rgb.blue 1 -ram %s -progress 1"
        % (otb_path, input_image_tif, output_image_jpg, ram_preview)
    )
    if not os.path.exists(output_image_jpg):
        print(cmd_convert_to_jpg)
        try:
            retcode = subprocess.check_call(cmd_convert_to_jpg, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Convert JPG failed =======================")
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "===================================================================="
            )
            sys.exit(1)
        else:
            print(
                "====================== Convert JPG done. ============================"
            )
            print(
                "===================================================================="
            )
    else:
        print("%s has already been converted." % output_image_jpg)


def rm_xml_files(target_directory):
    """
    Remove unnecessary xml files in the target directory.
    """
    xml_file_list = glob.glob(target_directory + "/*.xml")
    for xml_file in xml_file_list:
        os.remove(xml_file)


def create_original_image_jpg_preview_(
    otb_path, sorted_date_image_folder_path_tuple, output_path, ram_preview
):
    """This function create false color preview images in jpg format.
    The size of image is 10 time less than original images.
    It receives a list of images for one date,
    chooses three bands (band3, band4, band8),
    rescale each image,
    concatenate them to one tif,
    convert to jpg
    """
    date = sorted_date_image_folder_path_tuple[0]
    input_image_list = glob.glob(sorted_date_image_folder_path_tuple[1] + "/*.tif")
    target_image_list = [
        image for image in input_image_list if only_select_3_bands(image)
    ]
    output_image_list = []
    for image in target_image_list:
        tmp_output_path = os.path.join(output_path, "tmp", os.path.basename(image))
        rescale_to_1000_1000(otb_path, image, tmp_output_path, ram_preview)
        output_image_list.append(tmp_output_path)
    output_image_list.sort()
    output_image_tif = os.path.join(output_path, "tmp", date + "_falsecolor.tif")
    stack_images_to_tif(otb_path, output_image_list, output_image_tif, ram_preview)
    output_image_jpg = os.path.join(output_path, "Original", date + "_falsecolor.jpg")
    convert_to_jpg(otb_path, output_image_tif, output_image_jpg, ram_preview)
    # Remove xml files
    rm_xml_files(os.path.join(output_path, "Original"))

    # Convert to three-bands binary mask


def convert_mask_to_three_bands_binary(
    otb_path, mask_path_list, three_bands_binary_mask_path, ram_preview
):
    """Convert CLM, EDG and SAT masks to binary mask using OTB command
            0: Valid pixel
            1: Invalid pixel
    And concatenate them into one tif image with three bands.
    """
    command = (
        "%s/otbcli_BandMathX -il %s -out %s uint8 -exp '(im1b1>0?1:0);(im2b1==1?1:0);(im3b1==1?1:0)' -ram %s -progress 1"
        % (
            otb_path,
            " ".join(mask_path_list),
            three_bands_binary_mask_path,
            ram_preview,
        )
    )
    print(command)
    if not os.path.exists(three_bands_binary_mask_path):
        try:
            retcode = subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(
                "==================== Convert three_band_binary mask failed ======================="
            )
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "==================================================================================="
            )
            sys.exit(1)
        else:
            print(
                "=================== Convert three_band_binary mask done. =========================="
            )
            print(
                "==================================================================================="
            )
    else:
        print("%s has already existed." % three_bands_binary_mask_path)


def create_binary_mask_jpg_preview(
    otb_path, date, input_bmask, preview_output_path, ram_preview
):
    """This function receives the date, a binary mask tif (3 bands) and mask preview output path
    Process:
            rescale,
            convert to jpg.
    Color Mapping:
            R: CLM
            G: EDF
            B: SAT
    Output is one jpg review mask image for the date
    """
    """
	rescaled_output_mask_list = []
	for mask in input_mask_list:
		tmp_output_path = os.path.join(preview_output_path, "tmp", os.path.basename(mask))
		rescale_to_1000_1000(otb_path, mask, tmp_output_path)
		rescaled_output_mask_list.append(mask)
	output_mask_tif = os.path.join(preview_output_path, "tmp", date + "_mask.tif")
	stack_masks_to_tif(otb_path, rescaled_output_mask_list, output_mask_tif)
	output_mask_jpg = os.path.join(preview_output_path, "Mask", date + "_mask.jpg")
	cmd_convert_to_jpg = "%s/otbcli_DynamicConvert -in %s -out %s -quantile -channels grayscale -ram 1000 -progress 1" % (otb_path, output_mask_tif, output_mask_jpg)
	if not os.path.exists(output_mask_jpg):
		print(cmd_convert_to_jpg)
		os.system(cmd_convert_to_jpg)
	else:
		print("%s has already been converted." % output_mask_jpg)
	"""
    tmp_output_path = os.path.join(
        preview_output_path, "tmp", os.path.basename(input_bmask)
    )
    rescale_to_1000_1000(otb_path, input_bmask, tmp_output_path, ram_preview)
    output_mask_jpg = os.path.join(preview_output_path, "Mask", date + "_mask.jpg")
    cmd_convert_to_jpg = (
        "%s/otbcli_DynamicConvert -in %s -out %s -quantile -channels rgb -channels.rgb.red 1 -channels.rgb.green 2 -channels.rgb.blue 3 -ram %s -progress 1"
        % (otb_path, tmp_output_path, output_mask_jpg, ram_preview)
    )
    if not os.path.exists(output_mask_jpg):
        # print(cmd_convert_to_jpg)
        try:
            retcode = subprocess.check_call(cmd_convert_to_jpg, shell=True)
        except subprocess.CalledProcessError as e:
            print(
                "==================== Create binary mask preview failed ======================="
            )
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "=============================================================================="
            )
            sys.exit(1)
        else:
            print(
                "=================== Create binary mask preview done. =========================="
            )
            print(
                "==============================================================================="
            )
    else:
        print("%s has already been converted." % output_mask_jpg)
    rm_xml_files(os.path.join(preview_output_path, "Mask"))


# Sum masks for each tile.
def sum_mask_tif(
    otb_path, three_bands_binary_mask_list, sum_mask_tif_path, ram_preview
):
    """
    Create sum CLM mask for the tile.
    """
    exp = "'"
    for i in range(len(three_bands_binary_mask_list)):
        exp += "im%sb1 + " % str(i + 1)
    exp = exp[:-2] + "'"
    command = "%s/otbcli_BandMath -il %s -out %s uint8 -exp %s -ram %s -progress 1" % (
        otb_path,
        " ".join(three_bands_binary_mask_list),
        sum_mask_tif_path,
        exp,
        ram_preview,
    )
    if not os.path.exists(sum_mask_tif_path):
        try:
            retcode = subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print("==================== Create sum mask failed =======================")
            print("===================================================================")
            sys.exit(1)
        else:
            print(
                "=================== Create sum mask done. =========================="
            )
            print(
                "===================================================================="
            )
    else:
        print("%s has already created." % sum_mask_tif_path)


def extract_gapfilled_image_per_date(
    otb_path, date_list_file, input_gapfilled_image_tif, output_path, ram_processing
):
    """This function receive the number of total bands, synthetic dates list, gapfilled_image_tif,
    extract the 10 bands (band2, band3, band4, band5, band6, band7 band8, band8A, band11, band12) [10m && 20 m] for each dates
    return per date stacked images
    """
    nbands = 10  # TODO: global variable
    date_list = read_dates(date_list_file)

    for i in range(len(date_list)):
        date = date_list[i]
        # select the bands to be extracted
        cl_str = ""
        for add in range(nbands):
            bandi = i * nbands + (add + 1)
            cl_str += "Channel" + str(bandi) + " "
        date_image_tif = os.path.join(
            output_path, "t" + str(i) + "_" + date + "_gapfilled.tif"
        )
        cmd_extract_band = (
            "%s/otbcli_ExtractROI -in %s -out %s -mode standard -cl %s -ram %s -progress 1"
            % (
                otb_path,
                input_gapfilled_image_tif,
                date_image_tif,
                cl_str,
                ram_processing,
            )
        )
        if not os.path.exists(date_image_tif):
            try:
                retcode = subprocess.check_call(cmd_extract_band, shell=True)
            except subprocess.CalledProcessError as e:
                print(
                    "==================== Selecting bands failed ======================="
                )
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "==================================================================================="
                )
                sys.exit(1)
            else:
                print(
                    "=================== Selecting bands done. =========================="
                )
                print(
                    "==================================================================================="
                )


def create_gapfilled_image_jpg_preview(
    otb_path, date_list, input_gapfilled_image_tif, output_path, ram_preview
):
    """This function receive the number of total bands, synthetic dates list, gapfilled_image_tif,
    select the 3 bands (band3, band4, band8)
    return one false color preview image in jpg per date
    1. Rescale gapfilled image
    2. Select bands for each date.
    3. Convert output from 2 to JPG
    """
    rescaled_image_tif = os.path.join(
        output_path,
        "tmp",
        os.path.basename(os.path.splitext(input_gapfilled_image_tif)[0])
        + "_scaled.tif",
    )
    if not os.path.exists(rescaled_image_tif):
        rescale_to_1000_1000(
            otb_path, input_gapfilled_image_tif, rescaled_image_tif, ram_preview
        )
    for i in range(len(date_list)):
        date = date_list[i]
        # select the bands for preview image
        band3 = i * 10 + 2
        band4 = i * 10 + 3
        band8 = i * 10 + 7
        select_image_tif = os.path.join(output_path, "tmp", date + "_select.tif")
        cmd_select_band = (
            "%s/otbcli_ExtractROI -in %s -out %s -mode standard -cl Channel%s Channel%s Channel%s -ram %s -progress 1"
            % (
                otb_path,
                rescaled_image_tif,
                select_image_tif,
                str(band3),
                str(band4),
                str(band8),
                ram_preview,
            )
        )
        if not os.path.exists(select_image_tif):
            print(cmd_select_band)
            try:
                retcode = subprocess.check_call(cmd_select_band, shell=True)
            except subprocess.CalledProcessError as e:
                print(
                    "==================== Selecting bands failed ======================="
                )
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "==================================================================================="
                )
                sys.exit(1)
            else:
                print(
                    "=================== Selecting bands done. =========================="
                )
                print(
                    "==================================================================================="
                )
        output_image_jpg = os.path.join(
            output_path, "Gapfilled", date + "_gapfilled_falsecolor.jpg"
        )
        convert_to_jpg(otb_path, select_image_tif, output_image_jpg, ram_preview)
    rm_xml_files(os.path.join(output_path, "Gapfilled"))


def L2A_gapfilling_main_process(
    tile,
    in_dates_list_file,
    out_dates_list_file,
    input_path,
    output_path,
    otb_path,
    preview=False,
    preview_path=None,
    perdate=False,
    perdate_path=None,
    ram_processing=4000,
    ram_preview=1000,
):

    os.system("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8")
    ram_processing = str(ram_processing)
    ram_preview = str(ram_preview)
    tmp_path = os.path.join(output_path, "tmp")

    if preview:
        preview_original_path = os.path.join(preview_path, "Original")
        preview_mask_path = os.path.join(preview_path, "Mask")
        preview_gapfilled_path = os.path.join(preview_path, "Gapfilled")
        preview_tmp_path = os.path.join(preview_path, "tmp")
    if perdate:
        perdate_original_path = os.path.join(perdate_path, "Original")
        perdate_gapfilled_path = os.path.join(perdate_path, "Gapfilled")

    # Main processes for one tile.
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    if preview:
        if not os.path.exists(preview_path):
            os.mkdir(preview_path)

        if not os.path.exists(preview_original_path):
            os.mkdir(preview_original_path)

        if not os.path.exists(preview_mask_path):
            os.mkdir(preview_mask_path)

        if not os.path.exists(preview_gapfilled_path):
            os.mkdir(preview_gapfilled_path)

        if not os.path.exists(preview_tmp_path):
            os.mkdir(preview_tmp_path)

    if perdate:
        if not os.path.exists(perdate_path):
            os.mkdir(perdate_path)
        if not os.path.exists(perdate_original_path):
            os.mkdir(perdate_original_path)
        if not os.path.exists(perdate_gapfilled_path):
            os.mkdir(perdate_gapfilled_path)

    # 1. Read and select image files
    sorted_date_image_folder_path_tuple_list = sort_original_image_folders(
        input_path
    )  # Sort image file
    original_dates_file = write_original_dates(
        sorted_date_image_folder_path_tuple_list, output_path
    )  # write original_dates.txt to output_path
    all_sorted_target_image_paths_list = read_original_image_folders(
        sorted_date_image_folder_path_tuple_list, tmp_path
    )  # from each L2A folder, select 10 target bands tif files

    # 1.1 Create preview images for original images.
    if preview:
        print("====================================================================")
        print("================Create preview for original images=================")
        for (
            sorted_date_image_folder_path_tuple
        ) in sorted_date_image_folder_path_tuple_list:
            create_original_image_jpg_preview_(
                otb_path,
                sorted_date_image_folder_path_tuple,
                preview_path,
                ram_processing,
            )
        print("====================================================================")
        print("=================Preview for original images done. =================")
        print("====================================================================")

    # 2. Read and select mask files
    masks_list_for_stack = []
    if preview:
        masks_list_for_summask = []
    print("====================================================================")
    print("====================Convert masks to binary masks===================")
    for date_image_tuple in sorted_date_image_folder_path_tuple_list:
        date = date_image_tuple[0]
        image_folder_path = date_image_tuple[1]
        target_mask_list = read_original_masks(image_folder_path)

        # if not preview:
        binary_mask_path = os.path.join(tmp_path, date + "_binary_mask.tif")
        # Convert masks to one band binary masks
        convert_mask_to_binary(
            otb_path, target_mask_list, binary_mask_path, ram_processing
        )
        masks_list_for_stack.append(binary_mask_path)

        if preview or perdate:
            # Convert masks to three band binary masks
            three_bands_binary_mask_path = os.path.join(
                tmp_path, date + "_three_bands_binary_mask.tif"
            )
            convert_mask_to_three_bands_binary(
                otb_path, target_mask_list, three_bands_binary_mask_path, ram_preview
            )
            # Create binary masks jpg preview
            create_binary_mask_jpg_preview(
                otb_path, date, three_bands_binary_mask_path, preview_path, ram_preview
            )
            masks_list_for_summask.append(three_bands_binary_mask_path)

    print("===================== Mask Converting finished. ====================")
    print("====================================================================")

    if preview:
        # 2.1 Create sum mask file
        print("====================================================================")
        print("======================= Create sum mask file =======================")
        sum_mask_tif_path = os.path.join(preview_path, "%s_sum_mask.tif" % tile)
        sum_mask_tif(otb_path, masks_list_for_summask, sum_mask_tif_path, ram_preview)
        print("================= Creatation of sum mask finished. =================")
        print("====================================================================")

    # if not preview:
    # 3. Resample 20m resolution images to 10m resolution
    print("====================================================================")
    print("===================Start to resample r20m images====================")
    images_list_for_stack = resample_20_to_10(
        otb_path, all_sorted_target_image_paths_list, tmp_path, ram_processing
    )

    print("=========================Resampling finished========================")
    print("====================================================================")

    # 4. Stack image files (or masks files) to single tif file
    print("====================================================================")
    print("=======================print image list for stack===================")
    for image in images_list_for_stack:
        print(image)
    print("====================================================================")
    print("====================================================================")
    print("=======================Start to stack images========================")
    image_tif_path = os.path.join(output_path, "%s_Image.tif" % tile)
    stack_images_to_tif(otb_path, images_list_for_stack, image_tif_path, ram_processing)
    print("======================Image Stacking finished========================")
    print("=====================================================================")

    print("====================================================================")
    print("=======================print masks list for stack===================")
    for mask in masks_list_for_stack:
        print(mask)
    print("====================================================================")
    print("====================================================================")
    print("=======================Start to stack masks=========================")
    mask_tif_path = os.path.join(output_path, "%s_Mask.tif" % tile)
    stack_masks_to_tif(otb_path, masks_list_for_stack, mask_tif_path, ram_processing)
    print("======================Mask Stacking finished========================")
    print("====================================================================")
    print("\n")
    # ~ try:
    # ~ retcode = subprocess.check_call(["rm", '-r', input_path])
    # ~ except subprocess.CalledProcessError as e:
    # ~ print("================== Remove unzipped images failed.===================")
    # ~ print("Last command is:", e.cmd)
    # ~ print("Exit status: ", e.returncode)
    # ~ print("===================================================================================")
    # ~ sys.exit(1)
    # ~ else:
    # ~ print("=================== Remove unzipped images done. ==========================")
    # ~ print("===================================================================================")

    # 5. Gapfilling
    gapfilled_image_tif = os.path.join(output_path, "%s_GapFilled_Image.tif" % tile)
    print("====================================================================")
    print("=========================Gapfilling Process=========================")
    image_timeseries_gapfilling(
        otb_path,
        image_tif_path,
        mask_tif_path,
        gapfilled_image_tif,
        original_dates_file,
        out_dates_list_file,
        ram_processing,
    )
    print("========================Gapfilling finished=========================")
    print("====================================================================")

    if perdate:  # -- [Charlotte]
        # 6. Stacking images per date (gapfilled and not gapfilled)
        print(
            "==================================================================================="
        )
        print(
            "============Extract images per date for original and gapfilles images=============="
        )

        # 6.1 for original images
        stack_images_per_date_to_tif(
            otb_path,
            images_list_for_stack,
            tile,
            perdate_original_path,
            sorted_date_image_folder_path_tuple_list,
            masks_list_for_summask,
            True,
            ram_processing,
        )

        # 6.2 for gapfilled images
        if 0:
            extract_gapfilled_image_per_date(
                otb_path,
                out_dates_list_file,
                gapfilled_image_tif,
                perdate_gapfilled_path,
                ram_processing,
            )

        print(
            "=================================Extraction done==================================="
        )
        print(
            "==================================================================================="
        )

    if preview:  # -- not go for it
        # 8. Create previews for interpolated images according to synthetic dates
        print("====================================================================")
        print("================Create preview for gapfilled images=================")
        date_list = []
        output_tif = os.path.join(output_path, "%s_GapFilled_Image.tif" % tile)
        if os.path.exists(output_tif):
            with open(out_dates_list_file) as input_f:
                for line in input_f.readlines():
                    date_list.append(line.strip())
            create_gapfilled_image_jpg_preview(
                otb_path, date_list, output_tif, preview_path, ram_preview
            )
            print(
                "=================Preview for gapfilled images done. ================"
            )

            # ~ try:
            # ~ retcode = subprocess.check_call(["rm", '-r', preview_tmp_path])
            # ~ except subprocess.CalledProcessError as e:
            # ~ print("================== Remove temporary preview files failed.===================")
            # ~ print("Last command is:", e.cmd)
            # ~ print("Exit status: ", e.returncode)
            # ~ print("===================================================================================")
            # ~ sys.exit(1)
            # ~ else:
            # ~ print("=================== Remove temporary preview files done. ==========================")
            # ~ print("===================================================================================")
        else:
            print(
                "*** Gapfilling processing has not been done before creating previews. ***"
            )

        print("====================================================================")

    if 0:
        try:
            retcode = subprocess.check_call(["rm", "-r", input_path])
        except subprocess.CalledProcessError as e:
            print(
                "================== Remove unzipped images failed.==================="
            )
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "==================================================================================="
            )
            sys.exit(1)
        else:
            print(
                "=================== Remove unzipped images done. =========================="
            )
            print(
                "==================================================================================="
            )

        try:
            retcode = subprocess.check_call(["rm", "-r", tmp_path])
        except subprocess.CalledProcessError as e:
            print(
                "================== Remove temporary files failed.==================="
            )
            print("Last command is:", e.cmd)
            print("Exit status: ", e.returncode)
            print(
                "==================================================================================="
            )
            sys.exit(1)
        else:
            print(
                "=================== Remove temporary files done. =========================="
            )
            print(
                "==================================================================================="
            )


if __name__ == "__main__":
    # Global path:
    preview = True
    ram_processing = 4000
    ram_preview = 1000

    tile_names = ["T30UVU", "T30UWU"]

    for tile in tile_names:
        print("CHA: tile: ", tile)

        input_path = "/data/BreizhCrops/L2A_img/download_img/" + tile
        output_path = "/data/BreizhCrops/L2A_img/output/process/" + tile
        preview_path = "/data/BreizhCrops/L2A_img/output/preview/" + tile
        otb_path = "/home/cpelleti/OTB/OTB-7.0.0-Linux64/bin"

        # synthetic dates file path
        # ---- out_dates_list_file = "/home/zehui/summerproject/data/SeagateBlack/SeagateBlack/out_date_file.txt"
        in_dates_list_file = ""
        out_dates_list_file = ""

        L2A_gapfilling_main_process(
            tile,
            in_dates_list_file,
            out_dates_list_file,
            input_path,
            output_path,
            otb_path,
            preview,
            preview_path,
            ram_processing,
            ram_preview,
        )
