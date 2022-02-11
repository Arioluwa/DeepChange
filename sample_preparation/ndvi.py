<<<<<<< HEAD
import numpy as np
import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
import time
import argparse
import os 

parser = argparse.ArgumentParser(description="Plot NDVI")

parser.add_argument("-f", "--file", help="npz file path", required=True)
=======
from turtle import color
import numpy as np
import pandas as pd
import datetime
# import random
import matplotlib.pyplot as plt
import time
import argparse
import os

from pyparsing import alphas 

parser = argparse.ArgumentParser(description="Plot NDVI")

parser.add_argument("-f1", "--file1", help="npz file path", required=True)
parser.add_argument("-f2", "--file2", help="npz file path", required=True)
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
parser.add_argument("-g", "--gfdate", help="gfdate file path", required=True)
parser.add_argument("-o", "--output", help="output file path")

args = parser.parse_args()

<<<<<<< HEAD
f_path = args.file
=======
f_path = args.file1
f_path2 = args.file2
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
gfdate_path = args.gfdate
output_path = args.output

n_channel = 10

year = os.path.basename(f_path).split('_')[0]

<<<<<<< HEAD
=======
def convert_date_to_doy(gfdate_path):
    with open(gfdate_path, "r") as f:
        out_date_list = f.readlines()
    out_date_list = [x.strip() for x in out_date_list]
    out_date_list = [datetime.datetime.strptime(x, "%Y%m%d").timetuple().tm_yday for x in out_date_list]
    string_date_list = [str(x) for x in out_date_list]
    return string_date_list

# date of the year (doy) to be used as column name
date_label = convert_date_to_doy(gfdate_path)

    # len of time series
L = len(date_label)

label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]

>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
<<<<<<< HEAD
        polygon_ids = data["polygon_ids"]
        block_ids = data["block_id"]
    return X, y, polygon_ids, block_ids


def convert_date_to_doy(gfdate_path):
    with open(gfdate_path, "r") as f:
        out_date_list = f.read().splitlines()
    return [datetime.datetime.strptime(o, "%Y%m%d").timetuple().tm_yday for o in out_date_list]


def computeNDVI(X):
    """ """
=======
        # polygon_ids = data["polygon_ids"]
        # block_ids = data["block_id"]
    return X, y#, polygon_ids, block_ids


def computeNDVI(X):
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
    RED = np.array(X[:, :, 2]).astype(np.float16)
    NIR = np.array(X[:, :, 7]).astype(np.float16)

    with np.errstate(divide="ignore", invalid="ignore"):
        NDVI = np.where(NIR + RED != 0.0, (NIR - RED) / (NIR + RED), 0.0)
    np.seterr(divide="warn", invalid="warn")
    return NDVI.astype(np.float16)


<<<<<<< HEAD
def plot_chart(ndvi, title):
    fig, ax = plt.subplots(len(ndvi), sharex=True, figsize=(15, 40), constrained_layout=True)

    colorlist = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(19)
    ]

    for i in range(len(ndvi)):
        ax[i].plot(ndvi.iloc[i], color=colorlist[i])
        ax[i].set_title("Class: " + str(ndvi.index[i]))
        ax[i].set_ylabel("NDVI")
        fig.suptitle(title)
        plt.xlabel("Date")
        plt.savefig(os.path.join(output_path, title + year + ".png"))


def plot_NDVI():

    # load Npz data
    data_npz = load_npz(f_path)

    # date of the year (doy) to be used as column name
    date_label = convert_date_to_doy(gfdate_path)

    # len of time series
    L = len(date_label)

    # reshape the X into (N,T,D)
    X = np.reshape(data_npz[0], (data_npz[0].shape[0], L, n_channel))

    # compute NDVI on X == data_npz[0]
    NDVI = computeNDVI(X)

    # concatenate NDVI and class labels == data_npz[1]
    data = np.concatenate((NDVI, data_npz[1][:, None]), axis=1)
=======
def prepare_NDVI(f_path):
    X, y = load_npz(f_path)
    X_ = np.reshape(X, (X.shape[0], L, n_channel))
    NDVI = computeNDVI(X_)

    # concatenate NDVI and class
    data = np.concatenate((NDVI, y[:, None]), axis=1)

    # create a dataframe from data
    df = pd.DataFrame(data)

    # rename class labels index to 'code'
    df.columns = [*df.columns[:-1], "code"]

    # rename other columns to doy
    df.columns = [*date_label, "code"]

    #group all pixels by class label
    ndvi_mean = df.groupby(by="code").mean()

    ndvi_std = df.groupby(by="code").std()

    return ndvi_mean, ndvi_std


def plot_chart_ms(ndvi2018, ndvi2019, title):
    fig, ax = plt.subplots(len(ndvi2018[0]), sharex=True, figsize=(15, 40), constrained_layout=True)


    for i in range(len(ndvi2018[0])):
        ax[i].plot(ndvi2018[0].iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(ndvi2019[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].fill_between(range(L), ndvi2018[0].iloc[i] - ndvi2018[1].iloc[i], ndvi2018[0].iloc[i] + ndvi2018[1].iloc[i], alpha=0.1, color = '#D89CCB', label = 'std 2018')
        ax[i].fill_between(range(L), ndvi2019[0].iloc[i] - ndvi2019[1].iloc[i], ndvi2019[0].iloc[i] + ndvi2019[1].iloc[i], alpha=0.2, color='#CBCD9C', label='std 2019')
        ax[i].set_title(label[i])
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean_std.png"))
        # plt.savefig(os.path.join(output_path, title + year + ".png"))

def plot_chart_m(ndvi2018, ndvi2019, title):
    fig, ax = plt.subplots(len(ndvi2018[0]), sharex=True, figsize=(15, 40), constrained_layout=True)

    # colorlist = [
    #     "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    #     for i in range(19)
    # ]

    for i in range(len(ndvi2018[0])):
        ax[i].plot(ndvi2018[0].iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(ndvi2019[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        # ax[i].fill_between(range(L), ndvi2018[0].iloc[i] - ndvi2018[1].iloc[i], ndvi2018[0].iloc[i] + ndvi2018[1].iloc[i], alpha=0.1, color = '#BBD8CC', label = 'std 2018')
        # ax[i].fill_between(range(L), ndvi2019[0].iloc[i] - ndvi2019[1].iloc[i], ndvi2019[0].iloc[i] + ndvi2019[1].iloc[i], alpha=0.2, color='#CDA820', label='std 2019')
        ax[i].set_title(label[i])
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean.png"))

def plot_chart_y(ndvi2018, title):
    fig, ax = plt.subplots(len(ndvi2018[0]), sharex=True, figsize=(15, 40), constrained_layout=True)

    # colorlist = [
    #     "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    #     for i in range(19)
    # ]

    for i in range(len(ndvi2018[0])):
        ax[i].plot(ndvi2018[0].iloc[i], color='#5ab4ac', label='mean')
        # ax[i].plot(ndvi2019[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].fill_between(range(L), ndvi2018[0].iloc[i] - ndvi2018[1].iloc[i], ndvi2018[0].iloc[i] + ndvi2018[1].iloc[i], alpha=0.1, color = '#BBD8CC', label = 'std')
        # ax[i].fill_between(range(L), ndvi2019[0].iloc[i] - ndvi2019[1].iloc[i], ndvi2019[0].iloc[i] + ndvi2019[1].iloc[i], alpha=0.2, color='#CDA820', label='std 2019')
        ax[i].set_title(label[i])
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean_std_" + str(title.split()[-1]) + ".png"))

def prepare_NDVI_for_random_pixel(f_path):
    X, y = load_npz(f_path)
    X_ = np.reshape(X, (X.shape[0], L, n_channel))
    NDVI = computeNDVI(X_)

    # concatenate NDVI and class
    data = np.concatenate((NDVI, y[:, None]), axis=1)
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed

    # create a dataframe from data
    df = pd.DataFrame(data)

<<<<<<< HEAD
    # rename class labels index to 'code
=======
    # rename class labels index to 'code'
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
    df.columns = [*df.columns[:-1], "code"]

    # rename other columns to doy
    df.columns = [*date_label, "code"]

<<<<<<< HEAD
    # group all pixels by class label
    ndvi_mean = df.groupby("code").mean()

    ndvi_std = df.groupby("code").std()

    ########### Edit here ##############
    # using subplot
    # fig, ax = plt.subplots(len(ndvi_mean), sharex=True, figsize=(15, 40))

    # colorlist = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(19)]

    # for i in range(len(ndvi_mean)):
    #     ax[i].plot(ndvi_mean.iloc[i], color=colorlist[i])
    #     ax[i].set_title('Class: '+ str(ndvi_mean.index[i]))

    # plot each class mean NDVI
    plot_chart(ndvi_mean, "NDVI_mean")
    # plot each class std NDVI
    plot_chart(ndvi_std, "NDVI_std")

    # plot all the class mean NDVI
    (ndvi_mean.T).plot(kind="line", figsize=(35, 15))
    plt.title("NDVI_mean")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.savefig(os.path.join(output_path, "NDVI_mean_all_class" + year + ".png"))
    
=======
    random_sample = df.groupby("code").apply(lambda x: x.sample(n=1))

    random_sample.drop(columns=["code"], inplace=True)

    return random_sample


def plot_chart_random(ndvi2018, ndvi2019, title):
    fig, ax = plt.subplots(len(ndvi2018), sharex=True, figsize=(15, 40), constrained_layout=True)

    for i in range(len(ndvi2018)):
        ax[i].plot(ndvi2018.iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(ndvi2019.iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].set_title(label[i])
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_random_pixels.png"))

def plot_NDVI():
#    ndvi1 = prepare_NDVI(f_path)
#    ndvi2 = prepare_NDVI(f_path2)
     ndvi3 = prepare_NDVI_for_random_pixel(f_path)
     ndvi4 = prepare_NDVI_for_random_pixel(f_path2)

#    plot_chart_ms(ndvi1, ndvi2, "NDVI mean and standard deviation for 2018 and 2019")
#    plot_chart_m(ndvi1, ndvi2, "NDVI mean for 2018 and 2019")
#    plot_chart_y(ndvi1, "NDVI mean for 2018")
#    plot_chart_y(ndvi2, "NDVI mean for 2019")
     plot_chart_random(ndvi3, ndvi4, "Class NDVI for random pixels 2018 and 2019")

>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed

time_start = time.time()
plot_NDVI()
print("Time elapsed: ", time.time() - time_start)
<<<<<<< HEAD
=======


>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed
