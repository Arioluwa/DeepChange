#! /usr/bin/env python
# -*- coding:utf-8 -*-

from stacking import *
from generate_sqlite import *
import sys
import os
import glob
import optparse
import subprocess

# Global variables
_otb_path = "/share/projects/erasmus/deepchange/codebase/OTB-7.4.0-Linux64/bin"
# _l2a_processed_saving_path = "/media/cacao/SeagateRed/pepsL2A_processed_img"
# _peps_auth_path = ""
# _gapfilled_dates = "/media/petitjean/nfs-sits-archive/vault/S2_2018_Aus_backup/out_date_file.txt"
# _shp_file = "/media/petitjean/nfs-sits-archive/vault/S2_2018_Aus_backup/ref_data_with_urban_polygon/final_ref_data/VLUIS_GroundData_2017.shp"
# _shp_mask_folder = "/home/petitjean/Dropbox/workspace/summerprojectcodebase/data/tiles"
# _tiles_lookuptable = "/media/petitjean/nfs-sits-archive/vault/S2_2018_Aus_backup/tiles_lookuptable.csv"
# _field = "lc_id"

# --PREV: _gapfilled_dates = "/data/BreizhCrops/L2A_img/output/gapfilled_dates.txt"
_gapfilled_dates = "/share/projects/erasmus/deepchange/codebase/DeepChange/data_processing/gapfilled19_dates.txt"
_shp_file = "/share/projects/erasmus/deepchange/data/samples_shapefiles/samples_oso2019_T31TCJ.shp"
_shp_mask_folder = (
    "/share/projects/erasmus/deepchange/data/theiaL2A_zip_img/TileBorders"
)
_tiles_lookuptable = ""  # -- No need 
_field = "code"


class OptionParser(optparse.OptionParser):
    def check_required(self, opt):
        option = self.get_option(opt)
        # Assumes the option's 'default' is set to None!
        if getattr(self.values, option.dest) is None:
            self.error("%s option not supplied" % option)


# ==================
# parse command line
# ==================
if len(sys.argv) == 1:
    prog = os.path.basename(sys.argv[0])
    print("       " + sys.argv[0] + " [option]")
    print("     Aide: ", prog, "--help")
    print("         ou: ", prog, "  -h")
    print(
        "example command: python processingchain.py -a /media/cacao/SeagateBlack/pepsL2A_zip_img/54HWC -p FALSE"
    )
    print("or")
    print(
        "example command: python processingchain.py -d /media/cacao/SeagateBlack/pepsL2A_zip_img -l 54HWC 54HWD 54HWE -p FALSE"
    )
else:
    usage = " usage: %prog [options] "
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-a",
        "--atile",
        dest="atile",
        action="store",
        type="string",
        help="The L2A zip files for a given tile.",
    )
    parser.add_option(
        "-d",
        "--dir",
        dest="dir",
        action="store",
        type="string",
        help="The directory of L2A zip files for all tiles.",
    )
    parser.add_option(
        "-o",
        "--output_dir",
        dest="output_dir",
        action="store",
        type="string",
        help="The directory where the processed images are stored (the folder that will contain the 'pepsL2A_processed_img' folder).",
    )
    parser.add_option(
        "-l",
        "--tlist",
        dest="tlist",
        action="store",
        type="string",
        help="The tiles list to be processed.",
    )
    parser.add_option(
        "-p",
        "--preview",
        dest="preview",
        action="store",
        type="string",
        help="Create previews.",
        default="FALSE",
    )
    parser.add_option(
        "-t",
        "--perdate",
        dest="perdate",
        action="store",
        type="string",
        help="Create per date images.",
        default="FALSE",
    )
    ######### Shp and gapfilled dates###############
    parser.add_option(
        "-g",
        "--gfd",
        dest="_gapfilled_dates",
        action="store",
        type="string",
        help="Directory path to gapfilled dates.",
    )
    parser.add_option(
        "-s",
        "--shf",
        dest="_shp_file",
        action="store",
        type="string",
        help="Directory path to the sample shapefile.",
    )
    (options, args) = parser.parse_args()


""" Document for temporary experiment
if options.atile:
	if not os.path.exists(options.atile):
		print("The tiles list file %s does not exist" % options.atile)
		sys.exit(-1)
"""


if options.dir:
    if not os.path.exists(options.dir):
        print("The output directory %s does not exist" % options.dir)
        sys.exit(-1)


tile_folder_path = options.atile
tiles_dir_path = options.dir
perdate = options.perdate.lower() == "true"

if options.tlist != None:
    tile_list = options.tlist.split(" ")

preview = options.preview.lower() == "true"
if options.output_dir == None:
    _l2a_processed_saving_path = os.path.join(
        os.path.split(os.path.split(tile_folder_path)[0])[0], "theaiL2A_processed_img"
    )
else:
    # _l2a_processed_saving_path = os.path.join(options.output_dir, "pepsL2A_processed_img")
    _l2a_processed_saving_path = options.output_dir

ram_processing = 15000
ram_preview = 15000

if tile_folder_path:
    tile = os.path.basename(tile_folder_path)
    output_path = os.path.join(_l2a_processed_saving_path, tile)
    tmp_unzip_path = os.path.join(output_path, "tmp_unzip")
    original_dates_file = os.path.join(output_path, "original_dates.txt")
    # check where the program stops last time running
    stacked_mask = os.path.join(output_path, "%s_Mask.tif" % tile)
    # gapfilled_image = os.path.join(output_path, "%s_GapFilled_Image.tif" %tile)

    ########################### STEP 1: Unzip and gapfilling ############################
    # Case1: if stacking is not finished， the process goes from the unzipping
    if (not os.path.exists(stacked_mask)) or preview is True:

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if not os.path.exists(tmp_unzip_path):
            os.mkdir(tmp_unzip_path)
            # Unzip (zan shi fang zhi zai if xia mian) <-- "temporarily placed under if"
            print(
                "===================================================================="
            )
            print(
                "===================== Unzip processing ============================="
            )
            # not unzipping SRE (not corrected for slope) and ATB (athmospheric info from MAJA) files
            cmd = (
                "unzip -u %s/\*.zip -x *_SRE_*.tif *_ATB_*.tif *R2.tif *_IAB_*.tif *_MG2_*.tif -d %s"
                % (tile_folder_path, tmp_unzip_path)
            )
            try:
                retcode = subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError as e:
                print(
                    "======================= Unzip file failed. ======================="
                )
                print("Last command is:", e.cmd)
                print("Exit status: ", e.returncode)
                print(
                    "===================================================================="
                )
                # sys.exit(1)
            else:
                print(
                    "======================= Unzip file finished. ======================="
                )
                print(
                    "===================================================================="
                )
                print("\n")

        # Gapfilling process
        input_path = tmp_unzip_path
        otb_path = _otb_path

        if preview:
            preview_path = os.path.join(output_path, "preview")
        else:
            preview_path = None

        if perdate:
            perdate_path = os.path.join(output_path, "perdate")
        else:
            perdate_path = None

        # synthetic dates file path
        out_dates_list_file = _gapfilled_dates
        #out_dates_list_file = options._gapfilled_dates
        # out_dates_list_file = original_dates_file
        original_dates_file = os.path.join(output_path, "original_dates.txt")
        in_dates_list_file = original_dates_file

        # Main gapfilling process
        print("====================================================================")
        print("====================== Gapfilling process ==========================")
        L2A_gapfilling_main_process(
            tile,
            in_dates_list_file,
            out_dates_list_file,
            input_path,
            output_path,
            otb_path,
            preview,
            preview_path,
            perdate,
            perdate_path,
            ram_processing,
            ram_preview,
        )

    # Case2:  if the stacked has been finished, go directly to the gapfilling process, otherwise, go the unzip process
    if os.path.exists(stacked_mask) and preview is False:

        stacked_image = os.path.join(output_path, "%s_Image.tif" % tile)
        original_dates_file = os.path.join(output_path, "original_dates.txt")

        # Initial parameters
        otb_path = _otb_path
        out_dates_list_file = _gapfilled_dates
        #out_dates_list_file = options._gapfilled_dates
        # out_dates_list_file = original_dates_file
        in_dates_list_file = original_dates_file

        output_tif = os.path.join(output_path, "%s_GapFilled_Image.tif" % tile)
        # ram_processing = "1000"

        # gapfilling function
        image_timeseries_gapfilling(
            otb_path,
            stacked_image,
            stacked_mask,
            output_tif,
            original_dates_file,
            out_dates_list_file,
            ram_processing,
        )

        # remove temporary files
        if os.path.exists(tmp_unzip_path):
            try:
                retcode = subprocess.check_call(["rm", "-r", tmp_unzip_path])
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
        tmp_path = os.path.join(output_path, "tmp")
        if os.path.exists(tmp_path):
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

    ########################### STEP 2: Create SQLite file ############################
    otb_path = _otb_path
    img_file = os.path.join(output_path, "%s_Image.tif" % tile)
    #shp_file = options._shp_file
    shp_file = _shp_file
    shp_mask = os.path.join(_shp_mask_folder, "S2_tile_T31TCJ.shp")
    field = _field

    if perdate:
        print("====================================================================")
        print("====================== Create SQLite file ==========================")
        generate_sqlite_perdate(
            otb_path, perdate_path, shp_file, shp_mask, field, output_path
        )
        print("====================================================================")
        print("The processing of %s has been finished." % tile)
        print("====================================================================")
    else:
        output_sqlite_path = os.path.join(output_path, tile + "_sample_extract.sqlite")
        if not os.path.exists(output_sqlite_path):
            img_file = os.path.join(output_path, "%s_GapFilled_Image.tif" % tile)
            #shp_file = options._shp_file
            shp_file = _shp_file
            tile_lookup = _tiles_lookuptable
            field = _field
            print(
                "===================================================================="
            )
            print(
                "====================== Create SQLite file =========================="
            )
            generate_sqlite(
                otb_path, img_file, shp_file, shp_mask, tile_lookup, field, output_path
            )

        print("====================================================================")
        print("The processing of %s has been finished." % tile)
        print("====================================================================")

if tiles_dir_path:
    for tile in tile_list:
        tile_folder_path = os.path.join(tiles_dir_path, tile)
        if not os.path.exists(tile_folder_path):
            print(
                "========================= Tile error ==================================="
            )
            print(
                "Tile %s is not existing in the given folder %s"
                % (tile, tiles_dir_path)
            )
            print(
                "========================================================================"
            )
        else:
            tile = os.path.basename(tile_folder_path)
            output_path = os.path.join(_l2a_processed_saving_path, tile)
            tmp_unzip_path = os.path.join(output_path, "tmp_unzip")
            original_dates_file = os.path.join(output_path, "original_dates.txt")

            # check where the program stops last time running
            stacked_mask = os.path.join(output_path, "%s_Mask.tif" % tile)
            # gapfilled_image = os.path.join(output_path, "%s_GapFilled_Image.tif" %tile)

            ########################### STEP 1: Unzip and gapfilling ############################
            # Case1: if stacking is not finished， the process goes from the unzipping
            if (not os.path.exists(stacked_mask)) or preview is True:

                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                if not os.path.exists(tmp_unzip_path):
                    os.mkdir(tmp_unzip_path)

                    # Unzip (zan shi fang zhi zai "if" xia mian)
                    print(
                        "===================================================================="
                    )
                    print(
                        "===================== Unzip processing ============================="
                    )
                    # cmd = "unzip -u %s/\*.zip -d %s" % (tile_folder_path, tmp_unzip_path)
                    # not unzipping SRE (not corrected for slope) and ATB (athmospheric info from MAJA) files
                    cmd = (
                        "unzip -u %s/\*.zip -x *_SRE_*.tif *_ATB_*.tif *R2.tif *_IAB_*.tif *_MG2_*.tif -d %s"
                        % (tile_folder_path, tmp_unzip_path)
                    )
                    try:
                        retcode = subprocess.check_call(cmd, shell=True)
                    except subprocess.CalledProcessError as e:
                        print(
                            "======================= Unzip file failed. ======================="
                        )
                        print("Last command is:", e.cmd)
                        print("Exit status: ", e.returncode)
                        print(
                            "===================================================================="
                        )
                        # sys.exit(1)
                    else:
                        print(
                            "======================= Unzip file finished. ======================="
                        )
                        print(
                            "===================================================================="
                        )
                        print("\n")

                # Gapfilling process
                input_path = tmp_unzip_path
                otb_path = _otb_path

                if preview:
                    preview_path = os.path.join(output_path, "preview")
                else:
                    preview_path = None

                if perdate:
                    perdate_path = os.path.join(output_path, "perdate")
                else:
                    perdate_path = None

                # ram_processing = 1000
                # ram_preview = 1000

                # synthetic dates file path
                out_dates_list_file = _gapfilled_dates
                #out_dates_list_file = options._gapfilled_dates
                # out_dates_list_file = original_dates_file
                in_dates_list_file = original_dates_file

                # Main gapfilling process
                print(
                    "===================================================================="
                )
                print(
                    "====================== Gapfilling process =========================="
                )
                L2A_gapfilling_main_process(
                    tile,
                    in_dates_list_file,
                    out_dates_list_file,
                    input_path,
                    output_path,
                    otb_path,
                    preview,
                    preview_path,
                    perdate,
                    perdate_path,
                    ram_processing,
                    ram_preview,
                )

            # Case2:  if the stacked has been finished, go directly to the gapfilling process, otherwise, go the unzip process
            if os.path.exists(stacked_mask) and preview is False:
                stacked_image = os.path.join(output_path, "%s_Image.tif" % tile)
                original_dates_file = os.path.join(output_path, "original_dates.txt")

                # Initial parameters
                otb_path = _otb_path
                out_dates_list_file = _gapfilled_dates
                #out_dates_list_file = options._gapfilled_dates
                # out_dates_list_file = original_dates_file
                in_dates_list_file = original_dates_file
                output_tif = os.path.join(output_path, "%s_GapFilled_Image.tif" % tile)
                # ram_processing = "1000"

                # gapfilling function
                image_timeseries_gapfilling(
                    otb_path,
                    stacked_image,
                    stacked_mask,
                    output_tif,
                    original_dates_file,
                    out_dates_list_file,
                    ram_processing,
                )

                # remove temporary files
                if os.path.exists(tmp_unzip_path):
                    try:
                        retcode = subprocess.check_call(["rm", "-r", tmp_unzip_path])
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
                tmp_path = os.path.join(output_path, "tmp")
                if os.path.exists(tmp_path):
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

            ########################### STEP 2: Create SQLite file ############################
            otb_path = _otb_path
            otb_path = _otb_path
            img_file = os.path.join(output_path, "%s_Image.tif" % tile)
            #shp_file = options._shp_file
            shp_file = _shp_file
            shp_mask = os.path.join(_shp_mask_folder, "S2_tile_T31TCJ.shp")
            field = _field

            if perdate:
                print(
                    "===================================================================="
                )
                print(
                    "====================== Create SQLite file =========================="
                )
                generate_sqlite_perdate(
                    otb_path,
                    perdate_path,
                    shp_file,
                    shp_mask,
                    field,
                    output_path,
                    ram_processing,
                )
                print(
                    "===================================================================="
                )
                print("The processing of %s has been finished." % tile)
                print(
                    "===================================================================="
                )
            else:
                output_sqlite_path = os.path.join(
                    output_path, tile + "_sample_extract.sqlite"
                )
                if not os.path.exists(output_sqlite_path):
                    img_file = os.path.join(
                        output_path, "%s_GapFilled_Image.tif" % tile
                    )
                    #shp_file = options._shp_file
                    shp_file = _shp_file
                    tile_lookup = _tiles_lookuptable
                    field = _field
                    print(
                        "===================================================================="
                    )
                    print(
                        "====================== Create SQLite file =========================="
                    )
                    generate_sqlite(
                        otb_path,
                        img_file,
                        shp_file,
                        shp_mask,
                        tile_lookup,
                        field,
                        output_path,
                    )

                print(
                    "===================================================================="
                )
                print("The processing of %s has been finished." % tile)
                print(
                    "===================================================================="
                )
