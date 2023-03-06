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

from matplotlib import pyplot as plt, ticker as mticker, dates as mdates, transforms as mtransforms
import seaborn as sns
from compareForecasts import IceEdgeStatisticsFigure

import seaborn as sns


def main():
    sns.set_theme(context = "talk")
    # sns.despine()

    lead_time = sys.argv[1]
    grid = sys.argv[2]
    weights = sys.argv[3]

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

    
    files = [PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_OSISAF,  PATH_BARENTS]

    dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array

    ml_df = pd.read_csv(PATH_ML, index_col=0)

    ml_df = ml_df[ml_df.index.isin(dates)]

    ml_df['forecast_name'] = 'Deep learning'
    ml_df.index = pd.to_datetime(ml_df.index)

    fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5']

    mosaic_labels = ['c', 'd', '_', 'e', 'f']

    locator = mdates.YearLocator()
    fmt = mdates.DateFormatter(fmt='%b\n%Y')
    
    minor_locator = mdates.MonthLocator()
    minor_fmt = mdates.DateFormatter(fmt = '%b')

    # set common dates




    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()


    # Create figure classes
    if grid == 'nextsim':
        figname = f"{PATH_FIGURES}NS_P_ML_B_OSI_against_persistence.png"

    elif grid == 'amsr2':
        figname = f"{PATH_FIGURES}NS_P_ML_B_OSI_against_amsr2.png"

    else:
        exit('No valid grid supplied')

    figname = f"/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/poster_figs/performance.pdf"

    percentages = ['>=0', '>=10', '>=40', '>=70', '>=90', '=100']
    figsize = (19,19)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    axs = fig.subplot_mosaic('''
                             ac
                             ad
                             be
                             bf
                             ''')


    # Define list where forecasts are appended
    fetched_forecasts = [] 

    for forecast, name, label in zip(files, fnames, mosaic_labels):
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['NIIEE_mean'] = df.filter(regex=r'NIIEE_?[2-5]').mean(axis=1)

        df['forecast_name'] = name

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]


        if label != '_':
            print(name, label)
            df.index = pd.to_datetime(df.index)

            df['days_beat_niiee'] = ml_df['NIIEE_2'] < df['NIIEE_2']

            df_group = df.groupby(pd.Grouper(freq='M'))['days_beat_niiee'].sum()
            # df_size = df.groupby(['met_index']).size()
            df_size = df.groupby(pd.Grouper(freq='M')).size()

            df_days = df_group / df_size

            df_days.index = df_days.index.map(lambda x: x.replace(day = 1))
    
            if local_filename == 'barents':
                df_days.loc['2022-01':'2022-05'] = np.nan

            df_days = df_days.sort_index()
            print(df_days)

            sns.lineplot(data = df_days, x = df_days.index, y=df_days.values, marker = 'o', color='k', markeredgewidth='1.75', markersize = '14', mfc='red', mec ='k', ls = '--', ax = axs[label])
        
            axs[label].set_ylim(bottom = 0.4, top = 1.1)
            # axs[label].set_ylabel(f'Percentage of days with Deep learning NIIEE < {local_filename}')
        
            # axs[label].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc, nbins = 8))
            axs[label].set_yticks(np.arange(0.4, 1.2, 0.1))
            axs[label].set_yticklabels(['', '50%', '60%', '70%', '80%', '90%', '100%', ''])

            axs[label].grid(axis='x')
            # fig.autofmt_xdate()
            axs[label].set_xlim([datetime.date(2021, 12, 2), datetime.date(2022, 12, 31)])
            axs[label].xaxis.set_major_locator(locator)
            axs[label].xaxis.set_major_formatter(fmt)

            axs[label].xaxis.set_minor_locator(minor_locator)
            axs[label].xaxis.set_minor_formatter(minor_fmt)

            axs[label].set_title(f"% of days Deep learning beats {name}")

        
        if local_filename == 'barents':
            df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]


        fetched_forecasts.append(df[['NIIEE_2', 'NIIEE_mean', 'forecast_name', 'met_index']])
    

    fetched_dataframe = pd.concat(fetched_forecasts)

    x_label = 'met_index'
    hue_label = 'forecast_name'

    # Plot figure a)
    sns.boxplot(data = fetched_dataframe, 
        x = x_label, 
        y = 'NIIEE_2', 
        hue = hue_label,
        palette='deep',
        showmeans = True,
        meanprops={"marker": 'D', "markeredgecolor": 'black',
          "markerfacecolor": 'firebrick'},
        whis = [5,95],
        ax = axs['a'])
    
    axs['a'].set_ylim(top = 125)
    axs['a'].set_title(r'sic >= 10% contour')
    axs['a'].set_ylabel('')
    sns.move_legend(axs['a'], "best")
    axs['a'].legend_.set_title('Forecast product')

    # Plot figure b)
    sns.boxplot(data = fetched_dataframe, 
        x = x_label, 
        y = 'NIIEE_mean', 
        hue = hue_label,
        palette='deep',
        showmeans = True,
        meanprops={"marker": 'D', "markeredgecolor": 'black',
          "markerfacecolor": 'firebrick'},
        whis = [5,95],
        ax = axs['b'])
    
    axs['b'].set_ylim(top = 125)
    axs['b'].legend_.remove()
    axs['b'].set_ylabel('')
    axs['b'].set_title('Mean Normalized IIEE of (10-40, 40-70, 70-90, 90-100) contours')

    # Modify c,d,e,f days beat figures
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    for i in ['a', 'b', 'c', 'd', 'e', 'f']:
        axs[i].set_xlabel('')
        axs[i].text(0.0, 1.0, f"{i})", transform = axs[i].transAxes + trans, fontsize='medium', va='bottom')



    fig.supylabel('Ice edge displacement error [km] (Normalized IIEE)', ha='left')

    fig.suptitle('Model intercomparisson and percentage of days where Deep learning has lower Normalized IIEE (sic >= 10%)')
    
    fig.savefig(f"{figname}")


if __name__ == "__main__":
    main()