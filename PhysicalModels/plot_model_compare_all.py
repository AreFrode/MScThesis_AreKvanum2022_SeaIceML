import os
import sys
# sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt, ticker as mticker, dates as mdates, transforms as mtransforms
import seaborn as sns



def main():
    sns.set_theme(context = "talk")
    # sns.set_theme(context = "paper")
    
    # sns.despine()

    lead_time = [1, 2, 3]
    # grid = ['nextsim', 'amsr2']
    grid = ['nextsim']
    grid = [ele for ele in grid for _ in range(3)]

    weights = ['weights_08031256', 'weights_21021550', 'weights_09031047'] 

    # Define paths
    PATH_GENERAL = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/'

    PATH_FIGURES = f"{PATH_GENERAL}figures/thesis_figs/"

    # Create figure classes

    figname = f"{PATH_FIGURES}model_intercomparisson_all_nextsim.pdf"

    x_label = 'met_index'
    hue_label = 'forecast_name'

    mosaic_labels = ['a', 'b', 'c', 'd']
    categories = ['Open water', 'Very open drift ice', 'Open drift ice', 'Close drift ice', 'Very close drift ice', 'Fast ice']
    contours = ['10–30%', '40–60%', '70–80%', '90–100%']

    figsize = (16,18)

    fig = plt.figure(figsize = figsize, constrained_layout = True)

    inner1 = [
        ['a1'], 
        ['b1'], 
        ['c1'], 
        ['d1']
    ]
    inner2 = [
        ['a2'], 
        ['b2'], 
        ['c2'], 
        ['d2']
    ]
    inner3 = [
        ['a3'], 
        ['b3'], 
        ['c3'], 
        ['d3']
    ]

    inner4 = [
        ['a4'], 
        ['b4'], 
        ['c4'], 
        ['d4']
    ]
    inner5 = [
        ['a5'], 
        ['b5'], 
        ['c5'], 
        ['d5']
    ]
    inner6 = [
        ['a6'], 
        ['b6'], 
        ['c6'], 
        ['d6']
    ]

    nest1 = [
        [inner1, inner2, inner3],
    ]

    nest2 = [
        [inner4, inner5, inner6]
    ]

    outer = [
        [nest1]
    ]


    axs = fig.subplot_mosaic(outer)

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)
    
    # Read available statistics files

    # files = (PATH_NEXTSIM, PATH_OSISAF, PATH_ML, PATH_BARENTS, PATH_PERSISTENCE)
    # files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)
    # files = [PATH_NEXTSIM, PATH_ML, PATH_BARENTS]

    for id_, g, lead, model in zip(range(1, 7), grid, lead_time, weights):
        PATH_NEXTSIM = f"{PATH_GENERAL}{g}_grid/lead_time_{lead}/nextsim.csv"
        PATH_OSISAF = f"{PATH_GENERAL}{g}_grid/lead_time_{lead}/osisaf.csv"
        PATH_ML = f"{PATH_GENERAL}{g}_grid/lead_time_{lead}/{model}.csv"
        PATH_BARENTS = f"{PATH_GENERAL}{g}_grid/lead_time_{lead}/barents.csv"
        PATH_PERSISTENCE = f"{PATH_GENERAL}{g}_grid/lead_time_{lead}/persistence.csv"

        files = [PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_OSISAF,  PATH_BARENTS]

        meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
        seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

        months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()

        fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5']

        dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array

        ml_df = pd.read_csv(PATH_ML, index_col=0)

        ml_df = ml_df[ml_df.index.isin(dates)]

        ml_df['forecast_name'] = 'Deep learning'
        ml_df.index = pd.to_datetime(ml_df.index)
        
        # Define list where forecasts are appended
        fetched_forecasts = []


        for forecast, name in zip(files, fnames):
            local_filename = filename_regex.findall(forecast)[0]

            df = pd.read_csv(forecast, index_col = 0)
        
            # Find common dates
            df = df[df.index.isin(dates)]

            df['forecast_name'] = name

            for j, idx in zip(range(len(months) - 1), meteorological_seasons):
                df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]

            if local_filename == 'barents':
                df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]

            fetched_forecasts.append(df[[*[f'NIIEE_{contour}' for contour in range(2,6)], 'forecast_name', 'met_index']])

        fetched_dataframe = pd.concat(fetched_forecasts)


        for i, lab, cat, cont in zip(range(2, 6), mosaic_labels, categories, contours):
            sns.boxplot(data = fetched_dataframe, 
                x = x_label, 
                y = f'NIIEE_{i}', 
                hue = hue_label,
                palette='deep',
                showmeans = True,
                meanprops={"marker": 'D', "markeredgecolor": 'black',
                    "markerfacecolor": 'firebrick'},
                whis = [5,95],
                ax = axs[f"{lab}{id_}"])
    
            # axs[lab].set_ylim(top = 125)
            # axs[f"{lab}{id_}"].set_title(f'{cat} ({cont})')
            axs[f"{lab}{id_}"].set_ylabel('')
            axs[f"{lab}{id_}"].set_xlabel('')

            with sns.plotting_context('paper'):
                axs[f"{lab}{id_}"].legend()
            
            if not (lab == 'a' and id_ == 1):
                axs[f"{lab}{id_}"].legend_.remove()

            if lab in mosaic_labels[:3]:
                axs[f"{lab}{id_}"].set_xticklabels([])

            if id_ == 1:
                axs[f"{lab}{id_}"].set_ylabel(f'{cat} ({cont})')

            if lab == 'a':
                axs[f"{lab}{id_}"].set_title(f'{lead}-day lead time')
            
            if not lab in mosaic_labels[:3]:
                axs[f"{lab}{id_}"].xaxis.set_minor_locator(mticker.FixedLocator([1.5]))
                axs[f"{lab}{id_}"].xaxis.set_minor_formatter(mticker.FixedFormatter(['2022']))
            

    # exit()

    # sns.move_legend(axs['d3'], "upper left", bbox_to_anchor = (1,1))
    with sns.plotting_context('paper'):
        sns.move_legend(axs['a1'], "upper right")
        axs['a1'].legend_.set_title('Forecast product')

    # clim limits
    # axs['a'].set_ylim(top = 181)
    # axs['b'].set_ylim(top = 145)
    # axs['c'].set_ylim(top = 153)
    # axs['d'].set_ylim(top = 270)
    # axs['f'].set_ylim(bottom = -1, top = 30)

    # icechart limits
    # axs['a'].set_ylim(top = 60)
    # axs['b'].set_ylim(top = 30)
    # axs['c'].set_ylim(top = 42)
    # axs['d'].set_ylim(top = 100)
    # axs['f'].set_ylim(bottom = -1, top = 9)


    fig.supylabel('Ice edge displacement error [km] (Normalized IIEE)', ha='left')

    fig.suptitle('Normalized IIEE distribution for varying contours compared against Sea Ice Charts')
    
    fig.savefig(f"{figname}")


if __name__ == "__main__":
    main()