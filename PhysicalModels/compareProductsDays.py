import glob
import os
import sys
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import datetime

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt, ticker as mticker, dates as mdates
import seaborn as sns

import seaborn as sns

def my_format_function(x, pos=None):
    x = mdates.num2date(x)
    if pos == 0:
        fmt = '%b'
    else:
        fmt = '%b'
    label = x.strftime(fmt)
    return label


def main():
    sns.set_theme(context='talk')
    sns.despine()

    lead_time = sys.argv[1]
    grid = sys.argv[2]
    weights = sys.argv[3]
    contour = sys.argv[4]

    # Define paths
    PATH_NEXTSIM = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/nextsim.csv"
    PATH_OSISAF = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/osisaf.csv"
    PATH_ML = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/{weights}.csv"
    PATH_BARENTS = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/barents.csv"
    PATH_PERSISTENCE = f"/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/persistence.csv"
    PATH_FIGURES = "/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/poster_figs/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)
    
    # Read available statistics files

    # files = (PATH_NEXTSIM, PATH_OSISAF, PATH_ML, PATH_BARENTS, PATH_PERSISTENCE)
    # files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)
    # files = [PATH_NEXTSIM, PATH_ML, PATH_BARENTS]
    files = [PATH_ML, PATH_NEXTSIM, PATH_PERSISTENCE, PATH_OSISAF,  PATH_BARENTS]

    # set common dates

    dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array


    ml_df = pd.read_csv(PATH_ML, index_col=0)

    ml_df = ml_df[ml_df.index.isin(dates)]

    ml_df['Normalized_IIEE'] = ml_df[f'NIIEE_{int(contour)}']

    ml_df['forecast_name'] = 'Deep learning'
    ml_df.index = pd.to_datetime(ml_df.index)

    locator = mdates.YearLocator()
    fmt = mdates.DateFormatter(fmt='%b\n%Y')
    
    minor_locator = mdates.MonthLocator()
    minor_fmt = mdates.DateFormatter(fmt = '%b')

    for forecast in files[1:]:
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['Normalized_IIEE'] = df[f'NIIEE_{int(contour)}']

        df['forecast_name'] = local_filename

        df.index = pd.to_datetime(df.index)

        df['days_beat_niiee'] = ml_df['Normalized_IIEE'] < df['Normalized_IIEE']

        df_group = df.groupby(pd.Grouper(freq='M'))['days_beat_niiee'].sum()
        # df_size = df.groupby(['met_index']).size()
        df_size = df.groupby(pd.Grouper(freq='M')).size()

        df_days = df_group / df_size

        df_days.index = df_days.index.map(lambda x: x.replace(day = 1))
    
        if local_filename == 'barents':
            df_days.loc['2022-01':'2022-05'] = np.nan


        df_days = df_days.sort_index()
        print(df_days)

        plt.figure(figsize = (15,15))

        ax = sns.lineplot(data = df_days, x = df_days.index, y=df_days.values, marker = 'o', color='k', markeredgewidth='2.5', markersize = '22', mfc='red', mec ='k', ls = '--')
        
        ax.set_ylim(bottom = 0.4, top = 1.1)
        ax.set_ylabel(f'Percentage of days with Deep learning NIIEE < {local_filename}')
        
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(['', '50%', '60%', '70%', '80%', '90%', '100%', ''])

        ax.grid(axis='x')
        # fig.autofmt_xdate()
        ax.set_xlim([datetime.date(2021, 12, 2), datetime.date(2022, 12, 31)])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(fmt)

        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_minor_formatter(minor_fmt)
        # plt.gcf().autofmt_xdate()



        # ax.xaxis.set_major_locator(mticker.MaxNLocator(prune='both'))
        plt.savefig(f"{PATH_FIGURES}NIIEE_ML_fraction_{local_filename}.svg")


if __name__ == "__main__":
    main()