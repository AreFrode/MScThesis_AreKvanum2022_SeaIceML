# Extremely simple script where I punch in numbers manually to create heatmaps of loss based on unet-depth / learning rate and batch size

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt


def main():
    
    batchsize2 = np.array([[0.39261, 0.37780], 
                           [0.36445, 0.36766],
                           [0.47716, 0.54919],
                           [1.09546, 1.12673]])

    plt.figure()
    sns.heatmap(batchsize2, 
                annot = True, 
                fmt = ".5f", 
                xticklabels = ["512", "1024"], 
                yticklabels = ["0.01", "0.001", "0.0001", "0.00001"])

    plt.title("lr / unet-depth grid search for batch-size 2")
    plt.ylabel("Learning rate")
    plt.xlabel("Unet depth (# channels final conv.block)")
    plt.savefig("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/figures/grid_search_bs2.png")

    batchsize4 = np.array([[0.36403, 0.36862],
                           [0.55059, 0.61897],
                           [1.30086, np.nan]])

    plt.figure()
    sns.heatmap(batchsize4, 
                annot = True, 
                fmt = ".5f", 
                xticklabels = ["512", "1024"], 
                yticklabels = ["0.001", "0.0001", "0.00001"])

    plt.title("lr / unet-depth grid search for batch-size 4")
    plt.ylabel("Learning rate")
    plt.xlabel("Unet depth (# channels final conv.block)")
    plt.savefig("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/figures/grid_search_bs4.png")


if __name__ == "__main__":
    main()