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
parser.add_argument("-g", "--gfdate", help="gfdate file path", required=True)
parser.add_argument("-o", "--output", help="output file path")

args = parser.parse_args()

f_path = args.file
gfdate_path = args.gfdate
output_path = args.output

n_channel = 10

year = os.path.basename(f_path).split('_')[0]

def load_npz(file_path):
    """
    Load data from a .npz file
    """
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
        polygon_ids = data["polygon_ids"]
        block_ids = data["block_id"]
    return X, y, polygon_ids, block_ids


def convert_date_to_doy(gfdate_path):
    with open(gfdate_path, "r") as f:
        out_date_list = f.read().splitlines()
    return [datetime.datetime.strptime(o, "%Y%m%d").timetuple().tm_yday for o in out_date_list]


def computeNDVI(X):
    """ """
    RED = np.array(X[:, :, 2]).astype(np.float16)
    NIR = np.array(X[:, :, 7]).astype(np.float16)

    with np.errstate(divide="ignore", invalid="ignore"):
        NDVI = np.where(NIR + RED != 0.0, (NIR - RED) / (NIR + RED), 0.0)
    np.seterr(divide="warn", invalid="warn")
    return NDVI.astype(np.float16)


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

    # create a dataframe from data
    df = pd.DataFrame(data)

    # rename class labels index to 'code
    df.columns = [*df.columns[:-1], "code"]

    # rename other columns to doy
    df.columns = [*date_label, "code"]

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
    

time_start = time.time()
plot_NDVI()
print("Time elapsed: ", time.time() - time_start)
