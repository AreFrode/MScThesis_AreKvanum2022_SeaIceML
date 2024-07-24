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
from cmocean import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def season_sort(column):
    sorter = ['DJF', 'MAM', 'JJA', 'SON']
    correspondence = {sort: order for order, sort in enumerate(sorter)}
    return column.map(correspondence)

def fetch_forecast(file, common_dates, name):
    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()
    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    local_filename = filename_regex.findall(file)[0]

    df = pd.read_csv(file, index_col = 0)
        
    # Find common dates
    df = df[df.index.isin(common_dates)]

    df['forecast_name'] = name

    for j, idx in zip(range(len(months) - 1), meteorological_seasons):
        df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]

        if local_filename == 'barents':
            df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]

    
    return df[[*[f'NIIEE_{contour}' for contour in range(2,6)], 'forecast_name', 'met_index']]

def plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs, row, weights):
    start = 0

    if isinstance(weights, str):
        weights = [weights] * 3

    frames = np.empty((4, 3, 4))
    frames.fill(np.nan)

    for t, model in zip(lead_time, weights):
        PATH_ML = f"{PATH_GENERAL}nextsim_grid/lead_time_{t}/{model}.csv"

        df = fetch_forecast(PATH_ML, dates_lead[t - 1], 'Deep Learning').groupby('met_index').mean(numeric_only = True).sort_index(key = season_sort)

        if weights[0] == 'barents':
            start = 2
  
        frames[0,t-1,start:] = df['NIIEE_2'].to_numpy()
        frames[1,t-1,start:] = df['NIIEE_3'].to_numpy()
        frames[2,t-1,start:] = df['NIIEE_4'].to_numpy()
        frames[3,t-1,start:] = df['NIIEE_5'].to_numpy()

    frames = frames[:,::-1,:]

    for i, cat in enumerate(mosaic_labels):
        sns.heatmap(np.ma.masked_invalid(frames[i]), annot = True, cbar = False, ax = axs[f"{cat}{row}"], fmt = ".2f", vmin = 3.5, vmax = 20, cmap = cm.thermal)


        if row != 5:
            axs[f"{cat}{row}"].set_xticklabels([])
        
        else:
            axs[f"{cat}{row}"].set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])

        if cat != 'a':
            axs[f"{cat}{row}"].set_yticklabels([])

        else:
            axs[f"{cat}{row}"].set_yticklabels(['3-day', '2-day', '1-day'])
    


def main():
    sns.set_theme(context = "poster")
    # sns.set(font_scale = 3)
    # sns.set_theme(context = "paper")
    
    # sns.despine()

    lead_time = [1, 2, 3]
    # grid = ['nextsim', 'amsr2']
    grid = ['nextsim']
    grid = [ele for ele in grid for _ in range(3)]

    weights = ['weights_08031256', 'weights_21021550', 'weights_09031047']
    # models = ['']

    # Define paths
    PATH_GENERAL = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/'

    

    fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5']

    def PATH_NEXTSIM(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/nextsim.csv"
    def PATH_OSISAF(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/osisaf.csv"
    def PATH_ML(lead, model): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/{model}.csv"
    def PATH_BARENTS(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/barents.csv"
    def PATH_PERSISTENCE(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/persistence.csv"
    
    dates_lead = []

    files_lead1 = [PATH_NEXTSIM(1), PATH_PERSISTENCE(1), PATH_ML(1, weights[0]), PATH_OSISAF(1),  PATH_BARENTS(1)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead1], axis=1, join = 'inner').index.array)

    files_lead2 = [PATH_NEXTSIM(2), PATH_PERSISTENCE(2), PATH_ML(2, weights[1]), PATH_OSISAF(2),  PATH_BARENTS(2)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead2], axis=1, join = 'inner').index.array)

    files_lead3 = [PATH_NEXTSIM(3), PATH_PERSISTENCE(3), PATH_ML(3, weights[2]), PATH_OSISAF(3),  PATH_BARENTS(3)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead3], axis=1, join = 'inner').index.array)
    

    PATH_FIGURES = f"{PATH_GENERAL}figures/thesis_figs/"

    # Create figure classes

    figname = f"{PATH_FIGURES}model_intercomparisson_heatmap_nextsim.pdf"

    x_label = 'met_index'
    hue_label = 'forecast_name'

    mosaic_labels = ['a', 'b', 'c', 'd']
    categories = ['Open water', 'Very open drift ice', 'Open drift ice', 'Close drift ice', 'Very close drift ice', 'Fast ice']
    contours = ['10–30%', '40–60%', '70–80%', '90–100%']

    figsize = (18,19)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    subfigs = fig.subfigures(nrows = 5)

    inner1 = [
        ['a1', 'b1', 'c1', 'd1']
    ]
    inner2 = [
        ['a2', 'b2', 'c2', 'd2']
    ]
    inner3 = [
        ['a3', 'b3', 'c3', 'd3']
    ]
    inner4 = [
        ['a4', 'b4', 'c4', 'd4']
    ]
    inner5 = [
        ['a5', 'b5', 'c5', 'd5']
    ]
    '''
    outer = [
        [inner1],
        [inner2],
        [inner3],
        [inner4],
        [inner5]
    ]
    '''
    
    axs1 = subfigs[0].subplot_mosaic(inner1)
    axs2 = subfigs[1].subplot_mosaic(inner2)
    axs3 = subfigs[2].subplot_mosaic(inner3)
    axs4 = subfigs[3].subplot_mosaic(inner4)
    axs5 = subfigs[4].subplot_mosaic(inner5)


    # axs = fig.subplot_mosaic(outer)

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)


    plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs1, 1, 'osisaf')

    plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs2, 2, 'persistence')

    plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs3, 3, weights) 

    plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs4, 4, 'nextsim')

    plot_row(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs5, 5, 'barents')


    subfigs[0].suptitle('OSI SAF linear trend', y = .901)
    subfigs[1].suptitle('Persistence')
    subfigs[2].suptitle('Deep Learning')
    subfigs[3].suptitle('NeXtSIM')
    subfigs[4].suptitle('Barents-2.5')

    axs1["a1"].set_title(r'Ice edge = 10% sic', y = 1.2)
    axs1["b1"].set_title(r'Ice edge = 40% sic', y = 1.2)
    axs1["c1"].set_title(r'Ice edge = 70% sic', y = 1.2)
    axs1["d1"].set_title(r'Ice edge = 90% sic', y = 1.2)

    # fig.subplots_adjust(right = 0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    mapper = ScalarMappable(norm = Normalize(3.5, 20), cmap = cm.thermal)

    cb = fig.colorbar(mapper, ax = axs5['a5'], orientation = 'vertical', fraction = .05, pad = -.9, extend = 'max', ticks = [5, 10, 15, 20], format = mticker.FixedFormatter(['5', '10', '15', '> 20']))
    cb.outline.set_color('k')

    fig.supylabel('Ice edge displacement error [km]')

    fig.savefig(f"{figname}", dpi = 300)

if __name__ == "__main__":
    main()
