import os
import re
from datetime import datetime, date, timedelta


def sort_original_image_folders(input_path):
    """Read L2A images folders
       return a list of (date, image folder path) tuples sorted by date
    """
    image_files_list = os.listdir(input_path)
    list_of_tuples = []
    date_pattern = re.compile(r'\d{8}')
    for image_file in image_files_list:
        image_file_date = date_pattern.search(image_file).group(0)
        image_file_path = os.path.join(input_path, image_file)
        # list_of_tuples.append((image_file_date, image_file_path))
        list_of_tuples.append(image_file_date)
    list_of_tuples.sort() # sort by date
    # print to show sorted images
    # print("======================================================================")
    # print('\n')
    # print("Sorted image files by dates:")
    # print('\n')
    # for element in list_of_tuples:
    #     print("Date: %s, Abs path: %s" % (element[0], element[1]))
    # print("======================================================================")
    # print('\n')
    return list_of_tuples
path = "/share/projects/erasmus/deepchange/data/theiaL2A_zip_img/2019/"

out_date_list = sort_original_image_folders(path)
out_date_list_file = "gapfilled19_dates_.txt"
if not os.path.exists(out_date_list_file):
    with open(out_date_list_file, 'w') as f:
        print(out_date_list)
        for item in out_date_list:
            f.write("%s\n" % item)

############################## 2019 to 2018 ##############################
# convert YYYYMMDD to DOY
def convert_date_to_doy(date_str):
    """This function converts a date in the 
    format YYYYMMDD to a day of year (DOY)"""
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return date_obj.timetuple().tm_yday

# convert DOY to YYYYMMDD
def convert_doy_to_date(doy, year):
    """"This function converts a day of year (DOY) to a date string 
    in the format YYYYMMDD. It also recongnizes leap years"""
    doy.rjust(3 + len(doy), '0')
    date_obj = date(year, 1, 1) + timedelta(days=int(doy)-1)
    return date_obj.strftime('%Y%m%d')

# convert gapfilled (2019) to DOY
gapfilled_date_path = "/share/projects/erasmus/deepchange/codebase/DeepChange/data_processing/gapfilled19_dates.txt"
# gapfilled_date_path = "date.txt"

# read gapfilled dates
with open(gapfilled_date_path, 'r') as f:
    gapfilled_date_list = f.read().splitlines()

# create a list of DOYs using list comprehension
doy_list = [str(convert_date_to_doy(date_str)) for date_str in gapfilled_date_list]


################################
# create a list of dates using list comprehension
date_list = [str(convert_doy_to_date(doy_, 2018)) for doy_ in doy_list]


output_name = "gapfilled18_dates_.txt" # output file name

if not os.path.exists(output_name):
    with open(output_name, 'w') as f:
        print(date_list)
        for d in date_list:
            f.write(d + '\n')
