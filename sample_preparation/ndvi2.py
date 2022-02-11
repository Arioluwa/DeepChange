from turtle import color
import numpy as np
import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
import time
import argparse
import os

from pyparsing import alphas 

parser = argparse.ArgumentParser(description="Plot NDVI")

parser.add_argument("-f1", "--file1", help="npz file path", required=True)
parser.add_argument("-f2", "--file2", help="npz file path", required=True)
parser.add_argument("-g", "--gfdate", help="gfdate file path", required=True)
parser.add_argument("-o", "--output", help="output file path")

args = parser.parse_args()

f_path = args.file1
f_path2 = args.file2
gfdate_path = args.gfdate
output_path = args.output

n_channel = 10

year = os.path.basename(f_path).split('_')[0]

# def convert_date_to_doy(gfdate_path):
#     with open(gfdate_path, "r") as f:
#         out_date_list = f.read().splitlines()
#     return [datetime.datetime.strptime(o, "%Y%m%d").timetuple().tm_yday for o in out_date_list]

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

def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
        # polygon_ids = data["polygon_ids"]
        # block_ids = data["block_id"]
    return X, y#, polygon_ids, block_ids


def computeNDVI(X):
    RED = np.array(X[:, :, 2]).astype(np.float16)
    NIR = np.array(X[:, :, 7]).astype(np.float16)

    with np.errstate(divide="ignore", invalid="ignore"):
        NDVI = np.where(NIR + RED != 0.0, (NIR - RED) / (NIR + RED), 0.0)
    np.seterr(divide="warn", invalid="warn")
    return NDVI.astype(np.float16)


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

    # colorlist = [
    #     "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    #     for i in range(19)
    # ]

    for i in range(len(ndvi2018[0])):
        ax[i].plot(ndvi2018[0].iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(ndvi2019[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].fill_between(range(L), ndvi2018[0].iloc[i] - ndvi2018[1].iloc[i], ndvi2018[0].iloc[i] + ndvi2018[1].iloc[i], alpha=0.1, color = '#D89CCB', label = 'std 2018')
        ax[i].fill_between(range(L), ndvi2019[0].iloc[i] - ndvi2019[1].iloc[i], ndvi2019[0].iloc[i] + ndvi2019[1].iloc[i], alpha=0.2, color='#CBCD9C', label='std 2019')
        ax[i].set_title("Class: " + str(ndvi2018[0].index[i]))
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
        ax[i].set_title("Class: " + str(ndvi2018[0].index[i]))
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
        ax[i].set_title("Class: " + str(ndvi2018[0].index[i]))
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

    # create a dataframe from data
    df = pd.DataFrame(data)

    # rename class labels index to 'code'
    df.columns = [*df.columns[:-1], "code"]

    # rename other columns to doy
    df.columns = [*date_label, "code"]

    random_sample = df.groupby("code").apply(lambda x: x.sample(n=2))

    random_sample.drop(columns=["code"], inplace=True)

    return random_sample


def plot_chart_random(ndvi2018, ndvi2019, title):
<<<<<<< HEAD
    fig, ax = plt.subplots(2*len(ndvi2018), sharex=True, figsize=(15, 40), constrained_layout=True)
=======
    fig, ax = plt.subplots(len(ndvi2018), sharex=True, figsize=(15, 40), constrained_layout=True)
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed

    for i in range(len(ndvi2018)):
        ax[i].plot(ndvi2018.iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(ndvi2019.iloc[i], color= '#7fbf7b', linestyle='--', label='2019')

        ax[i].set_title("Class: " + str(ndvi2018.index[i]))
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_random_pixels.png"))

def plot_NDVI():
<<<<<<< HEAD
#    ndvi1 = prepare_NDVI(f_path)
#    ndvi2 = prepare_NDVI(f_path2)
    ndvi1 = prepare_NDVI_for_random_pixel(f_path)
    ndvi2 = prepare_NDVI_for_random_pixel(f_path2)

#    plot_chart_ms(ndvi1, ndvi2, "NDVI mean and standard deviation for 2018 and 2019")
#    plot_chart_m(ndvi1, ndvi2, "NDVI mean for 2018 and 2019")
#    plot_chart_y(ndvi1, "NDVI mean for 2018")
#    plot_chart_y(ndvi2, "NDVI mean for 2019")
    plot_chart_random(ndvi1, ndvi2, "Class NDVI for random pixels 2018 and 2019")
=======
    ndvi1 = prepare_NDVI(f_path)
    ndvi2 = prepare_NDVI(f_path2)
    ndvi3 = prepare_NDVI_for_random_pixel(f_path)
    ndvi4 = prepare_NDVI_for_random_pixel(f_path2)

    plot_chart_ms(ndvi1, ndvi2, "NDVI mean and standard deviation for 2018 and 2019")
    plot_chart_m(ndvi1, ndvi2, "NDVI mean for 2018 and 2019")
    plot_chart_y(ndvi1, "NDVI mean for 2018")
    plot_chart_y(ndvi2, "NDVI mean for 2019")
    plot_chart_random(ndvi3, ndvi4, "Class NDVI for random pixels 2018 and 2019")
>>>>>>> 1d364dbd00097303b2fd14b44f76d0b1770c0eed


time_start = time.time()
plot_NDVI()
print("Time elapsed: ", time.time() - time_start)

