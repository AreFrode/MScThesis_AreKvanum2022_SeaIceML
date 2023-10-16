import glob
import os
import sys
# sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import datetime

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt, ticker as mticker, dates as mdates, transforms as mtransforms
import seaborn as sns

import seaborn as sns


def main():
    sns.set_theme(context = "talk")
    # sns.despine()

    lead_time = sys.argv[1]
    grid = 'amsr2'
    weights = sys.argv[2]

    # Define paths
    PATH_NEXTSIM = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/nextsim.csv"
    PATH_OSISAF = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/osisaf.csv"
    PATH_ML = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/{weights}.csv"
    PATH_BARENTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/barents.csv"
    PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/persistence.csv"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/figures/thesis_figs/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)
    
    # Read available statistics files

    # files = (PATH_NEXTSIM, PATH_OSISAF, PATH_ML, PATH_BARENTS, PATH_PERSISTENCE)
    # files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)
    # files = [PATH_NEXTSIM, PATH_ML, PATH_BARENTS]

    
    files = [PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_OSISAF,  PATH_BARENTS]

    dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array

    ml_df = pd.read_csv(PATH_ML, index_col=0)

    ml_df = ml_df[ml_df.index.isin(dates)]

    ml_df['forecast_name'] = 'Deep learning'
    ml_df.index = pd.to_datetime(ml_df.index)

    fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5']


    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()


    # Create figure classes

    figname = f"{PATH_FIGURES}model_intercomparisson_amsr2.pdf"

    # Define list where forecasts are appended
    fetched_forecasts = [] 

    for forecast, name in zip(files, fnames):
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['forecast_name'] = name

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        
        if local_filename == 'barents':
            df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]


        fetched_forecasts.append(df[[*[f'NIIEE_{i}' for i in range(1,7)], 'forecast_name', 'met_index']])
    

    fetched_dataframe = pd.concat(fetched_forecasts)

    x_label = 'met_index'
    hue_label = 'forecast_name'

    # Plot figure a)
    mosaic_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    categories = ['Open water', 'Very open drift ice', 'Open drift ice', 'Close drift ice', 'Very close drift ice', 'Fast ice']
    contours = ['<10%', '10–30%', '40–60%', '70–80%', '90–100%', '100%']


    figsize = (14,14)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    axs = fig.subplot_mosaic('''
                             ab
                             cd
                             ef
                             ''')

    for i, lab, cat, cont in zip(range(1, 7), mosaic_labels, categories, contours):
        sns.boxplot(data = fetched_dataframe, 
            x = x_label, 
            y = f'NIIEE_{i}', 
            hue = hue_label,
            palette='deep',
            showmeans = True,
            meanprops={"marker": 'D', "markeredgecolor": 'black',
                "markerfacecolor": 'firebrick'},
            whis = [5,95],
            ax = axs[lab])
    
        # axs[lab].set_ylim(top = 125)
        axs[lab].set_title(f'{cat} ({cont})')
        axs[lab].set_ylabel('')
        axs[lab].set_xlabel('')

        if lab in ['a', 'b', 'c', 'e', 'f']:
            axs[lab].legend_.remove()

        if lab in mosaic_labels[:4]:
            axs[lab].set_xticklabels([])
    
        if not lab in mosaic_labels[:4]:
            axs[lab].xaxis.set_minor_locator(mticker.FixedLocator([1.5]))
            axs[lab].xaxis.set_minor_formatter(mticker.FixedFormatter(['2022']))

    # exit()

    sns.move_legend(axs['d'], "upper left", bbox_to_anchor = (1,1))
    axs['d'].legend_.set_title('Forecast product')

    axs['a'].set_ylim(top = 181)
    axs['b'].set_ylim(top = 145)
    axs['c'].set_ylim(top = 153)
    axs['d'].set_ylim(top = 270)
    # axs['f'].set_ylim(bottom = -1, top = 30)



    fig.supylabel('Ice edge displacement error [km] (Normalized IIEE)', ha='left')

    fig.suptitle('Normalized IIEE distribution for varying contours compared against AMSR2')
    
    fig.savefig(f"{figname}")


    fig = plt.figure(figsize = figsize, constrained_layout = True)

    ax = fig.add_subplot()

    fetched_dataframe.loc[fetched_dataframe['forecast_name'] == 'Deep learning', 'NIIEE_1'] = pd.read_csv(files[2], index_col = 0)['niiee_no_1contour']


    sns.boxplot(data = fetched_dataframe, 
            x = x_label, 
            y = f'NIIEE_1', 
            hue = hue_label,
            palette='deep',
            showmeans = True,
            meanprops={"marker": 'D', "markeredgecolor": 'black',
                "markerfacecolor": 'firebrick'},
            whis = [5,95],
            ax = ax)

    ax.set_title(f'Normalized IIEE distribution for {contours[0]} contour with same contour removed')
    ax.set_ylabel('Ice edge displacement error [km]', ha = 'left')

    ax.xaxis.set_minor_locator(mticker.FixedLocator([1.5]))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(['2022']))

    sns.move_legend(ax, "center left", bbox_to_anchor = (1,0.5))
    ax.legend_.set_title('Forecast product')

    ax.set_ylim(top = 181)
    
    fig.savefig(f"{PATH_FIGURES}removed_1contour_amsr2.pdf", dpi=300)


if __name__ == "__main__":
    main()
