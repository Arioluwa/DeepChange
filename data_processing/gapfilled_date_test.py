import re
import os 

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
out_date_list_file = "gapfilled19_dates.txt"
if not os.path.exists(out_date_list_file):
    with open(out_date_list_file, 'w') as f:
        print(out_date_list)
        for item in out_date_list:
            f.write("%s\n" % item)
