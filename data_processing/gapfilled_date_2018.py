import os
from datetime import datetime, date, timedelta

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


output_name = "gapfilled18_dates.txt" # output file name

if not os.path.exists(output_name):
    with open(output_name, 'w') as f:
        print(date_list)
        for d in date_list:
            f.write(d + '\n')
