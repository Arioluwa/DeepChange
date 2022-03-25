from turtle import color
import numpy as np
import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
import time
import argparse
import os

##########################
# Data analyse
# Mean NDVI for each class, comparing for both years and individual.
# Read the two npz files for both years and the gapfilled date,
# compute the NDVI after the data has been reshaped into (N-Number of pixels, T-Time, C-Channel)
##########################

parser = argparse.ArgumentParser(description="Plot NDVI")

parser.add_argument("-f1", "--source", help="npz file path", required=True)
parser.add_argument("-f2", "--target", help="npz file path", required=True)
parser.add_argument("-g", "--gfdate", help="gfdate file path", required=True)
parser.add_argument("-o", "--output", help="output file path")

args = parser.parse_args()

source_path = args.source
target_path = args.target
gfdate_path = args.gfdate
output_path = args.output

n_channel = 10

year = os.path.basename(f_path).split('_')[0]

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
    """
    Input: A reshaped data of (X, T, C)
    """
    RED = np.array(X[:, :, 2]).astype(np.float16)
    NIR = np.array(X[:, :, 7]).astype(np.float16)

    with np.errstate(divide="ignore", invalid="ignore"):
        NDVI = np.where(NIR + RED != 0.0, (NIR - RED) / (NIR + RED), 0.0)
    np.seterr(divide="warn", invalid="warn")
    return NDVI.astype(np.float16)


def prepare_NDVI(f_path):
    """
    Load and reshape the data
    Input: path to the npz file
        Compute NDVI
    Output: NDVI mean and std groupby dataframe
    
    """
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


def plot_mean_std(source_ndvi, target_ndvi, title):
    """
    Plot a mean and std chart for both source and target year.
    
    """
    # Number of subplots according to the length of the classes
    fig, ax = plt.subplots(len(source_ndvi[0]), sharex=True, figsize=(15, 40), constrained_layout=True)
    
    # for each subplot, plot the mean and std of each class
    for i in range(len(source_ndvi[0])):
        ax[i].plot(source_ndvi[0].iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(target_ndvi[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].fill_between(range(L), source_ndvi[0].iloc[i] - source_ndvi[1].iloc[i], source_ndvi[0].iloc[i] + source_ndvi[1].iloc[i], alpha=0.1, color = '#D89CCB', label = 'std 2018')
        ax[i].fill_between(range(L), target_ndvi[0].iloc[i] - target_ndvi[1].iloc[i], target_ndvi[0].iloc[i] + target_ndvi[1].iloc[i], alpha=0.2, color='#CBCD9C', label='std 2019')
        ax[i].set_title("Class: " + str(source_ndvi[0].index[i]))
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean_std.png"))
        # plt.savefig(os.path.join(output_path, title + year + ".png"))

def plot_mean_alone(source_ndvi, target_ndvi, title):
    """
    Plot only the mean for both years
    """
    fig, ax = plt.subplots(len(source_ndvi[0]), sharex=True, figsize=(15, 40), constrained_layout=True)

    for i in range(len(source_ndvi[0])):
        ax[i].plot(source_ndvi[0].iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(target_ndvi[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        # ax[i].fill_between(range(L), source_ndvi[0].iloc[i] - source_ndvi[1].iloc[i], source_ndvi[0].iloc[i] + source_ndvi[1].iloc[i], alpha=0.1, color = '#BBD8CC', label = 'std 2018')
        # ax[i].fill_between(range(L), target_ndvi[0].iloc[i] - target_ndvi[1].iloc[i], target_ndvi[0].iloc[i] + target_ndvi[1].iloc[i], alpha=0.2, color='#CDA820', label='std 2019')
        ax[i].set_title("Class: " + str(source_ndvi[0].index[i]))
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean.png"))

def plot_single_mean_std(ndvi_, title):
    """
    Plot the mean and std for a single year
    """
    fig, ax = plt.subplots(len(ndvi_[0]), sharex=True, figsize=(15, 40), constrained_layout=True)

    for i in range(len(ndvi_[0])):
        ax[i].plot(ndvi_[0].iloc[i], color='#5ab4ac', label='mean')
        # ax[i].plot(target_ndvi[0].iloc[i], color= '#7fbf7b', linestyle='--', label='2019')
        ax[i].fill_between(range(L), ndvi_[0].iloc[i] - ndvi_[1].iloc[i], ndvi_[0].iloc[i] + ndvi_[1].iloc[i], alpha=0.1, color = '#BBD8CC', label = 'std')
        # ax[i].fill_between(range(L), target_ndvi[0].iloc[i] - target_ndvi[1].iloc[i], target_ndvi[0].iloc[i] + target_ndvi[1].iloc[i], alpha=0.2, color='#CDA820', label='std 2019')
        ax[i].set_title("Class: " + str(ndvi_[0].index[i]))
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_mean_std_" + str(title.split()[-1]) + ".png"))

def prepare_NDVI_for_random_pixel(f_path):
    """
    Select some around pixels for each class and plot the NDVI
    """
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


def plot_random_mean_std(source_ndvi, target_ndvi, title):
    """
    Plot mean and std for random pixels representing each class.
    """
    
    fig, ax = plt.subplots(len(source_ndvi), sharex=True, figsize=(15, 40), constrained_layout=True)

    for i in range(len(source_ndvi)):
        ax[i].plot(source_ndvi.iloc[i], color='#5ab4ac', label='2018')
        ax[i].plot(target_ndvi.iloc[i], color= '#7fbf7b', linestyle='--', label='2019')

        ax[i].set_title("Class: " + str(source_ndvi.index[i]))
        ax[i].set_ylabel("NDVI")
        ax[i].legend()
        fig.suptitle(title)
        plt.xlabel("Date")
        fig.savefig(os.path.join(output_path, "NDVI_random_pixels.png"))

def plot_NDVI():
    """ 
    Calls all the functions above.
    """
    source_ = prepare_NDVI(source_path)
    target_ = prepare_NDVI(target_path)
    source_random_ = prepare_NDVI_for_random_pixel(source_path)
    target_random_ = prepare_NDVI_for_random_pixel(target_path)

    plot_mean_std(source_, target_, "NDVI mean and standard deviation for 2018 and 2019")
    plot_mean_alone(source_, target_, "NDVI mean for 2018 and 2019")
    plot_single_mean_std(source_, "NDVI mean for 2018")
    plot_single_mean_std(target_, "NDVI mean for 2019")
    plot_random_mean_std(source_random_, target_random_, "Class NDVI for random pixels 2018 and 2019")


time_start = time.time()
plot_NDVI()
print("Time elapsed: ", time.time() - time_start)

