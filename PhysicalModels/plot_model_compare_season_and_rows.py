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

def fetch_forecast(file, common_dates):
    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()
    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    local_filename = filename_regex.findall(file)[0]

    df = pd.read_csv(file, index_col = 0)
        
    # Find common dates
    df = df[df.index.isin(common_dates)]

    for j, idx in zip(range(len(months) - 1), meteorological_seasons):
        df.loc[(df.index >= months[j]) & (df.index < months[j+1]), 'met_index'] = seasonal_names[idx]

        if local_filename == 'barents':
            df.loc[(df['met_index'] == 'DJF') | (df['met_index'] == 'MAM') , ['NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']] = np.nan
    
    return df[[*[f'NIIEE_{contour}' for contour in range(2,6)], 'met_index']]

def plot_col(lead_time, PATH_GENERAL, dates_lead, mosaic_labels, axs, row, weights):
    start = 0

    frames = np.zeros((4, 6, 4))
    frames.fill(np.nan)

    for t, model in enumerate(weights):
        print(model)
        PATH_ML = f"{PATH_GENERAL}nextsim_grid/lead_time_{lead_time}/{model}.csv"

        df = fetch_forecast(PATH_ML, dates_lead).groupby('met_index').mean(numeric_only = True).sort_index(key = season_sort)
        print(df)
        

        if weights[0] == 'barents':
            start = 2
  
        frames[0, t, start:] = df['NIIEE_2'].to_numpy()
        frames[1, t, start:] = df['NIIEE_3'].to_numpy()
        frames[2, t, start:] = df['NIIEE_4'].to_numpy()
        frames[3, t, start:] = df['NIIEE_5'].to_numpy()


    for i in range(4):
        axs[i][f"{mosaic_labels}{i+1}"].plot([1,2,3,4], frames[i].T, '-o', label = ['Deep learning', 'Persistence', 'OSI SAF linear trend', 'Freedrift', 'NeXtSIM', 'Barents-2.5'])
        # sns.heatmap(np.ma.masked_invalid(frames[i]), annot = True, cbar = False, ax = axs[f"{cat}{row}"], fmt = ".2f", vmin = 3.5, vmax = 20, cmap = cm.thermal)

        axs[i][f"{mosaic_labels}{i+1}"].yaxis.set_major_formatter(mticker.FormatStrFormatter('%2d'))

        if i != 3:
            axs[i][f"{mosaic_labels}{i+1}"].set_xticks([1,2,3,4])
            axs[i][f"{mosaic_labels}{i+1}"].set_xticklabels([])
        
        else:
            axs[i][f"{mosaic_labels}{i+1}"].set_xticks([1,2,3,4])
            axs[i][f"{mosaic_labels}{i+1}"].set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])

        # if mosaic_labels != 'a':
            # axs[i][f"{mosaic_labels}{i+1}"].set_yticklabels([])

        # else:
            # axs[f"{cat}{row}"].set_yticklabels(['3-day', '2-day', '1-day'])


def main():
    fstyle_dict = {'axes.linewidth': 2.5,
                   'grid.linewidth': 2,
                   'lines.linewidth': 3.0,
                   'lines.markersize': 12,
                   'patch.linewidth': 2,
                   'xtick.major.width': 2.5,
                   'ytick.major.width': 2.5,
                   'xtick.minor.width': 2,
                   'ytick.minor.width': 2,
                   'xtick.major.size': 12,
                   'ytick.major.size': 12,
                   'xtick.minor.size': 8,
                   'ytick.minor.size': 8,
                   'font.size': 24,
                   'axes.labelsize': 24,
                   'axes.titlesize': 1.2*24,
                   'xtick.labelsize': 1.2*22,
                   'ytick.labelsize': 1.2*22,
                   'legend.fontsize': 1.1*22,
                   'legend.title_fontsize': 1.1*24}
    sns.set_theme(context = "poster", rc = fstyle_dict)

    '''
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    '''
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

    

    fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5', 'Freedrift']

    def PATH_NEXTSIM(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/nextsim.csv"
    def PATH_OSISAF(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/osisaf.csv"
    def PATH_ML(lead, model): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/{model}.csv"
    def PATH_BARENTS(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/barents.csv"
    def PATH_PERSISTENCE(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/persistence.csv"
    def PATH_FREEDRIFT(lead): return f"{PATH_GENERAL}nextsim_grid/lead_time_{lead}/freedrift.csv"
    
    dates_lead = []

    files_lead1 = [PATH_NEXTSIM(1), PATH_PERSISTENCE(1), PATH_ML(1, weights[0]), PATH_OSISAF(1),  PATH_BARENTS(1), PATH_FREEDRIFT(1)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead1], axis=1, join = 'inner').index.array)

    files_lead2 = [PATH_NEXTSIM(2), PATH_PERSISTENCE(2), PATH_ML(2, weights[1]), PATH_OSISAF(2),  PATH_BARENTS(2), PATH_FREEDRIFT(2)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead2], axis=1, join = 'inner').index.array)

    files_lead3 = [PATH_NEXTSIM(3), PATH_PERSISTENCE(3), PATH_ML(3, weights[2]), PATH_OSISAF(3),  PATH_BARENTS(3), PATH_FREEDRIFT(3)]
    dates_lead.append(pd.concat([pd.read_csv(file, index_col = 0) for file in files_lead3], axis=1, join = 'inner').index.array)


    PATH_FIGURES = f"{PATH_GENERAL}figures/thesis_figs/"

    # Create figure classes

    figname = f"{PATH_FIGURES}model_intercomparisson_seasonrows_nextsim.pdf"

    x_label = 'met_index'
    hue_label = 'forecast_name'

    mosaic_labels = ['a', 'b', 'c', 'd']
    categories = ['Open water', 'Very open drift ice', 'Open drift ice', 'Close drift ice', 'Very close drift ice', 'Fast ice']
    contours = ['10–30%', '40–60%', '70–80%', '90–100%']

    figsize = (24,20)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    subfigs = fig.subfigures(nrows = 5)

    inner1 = [
        ['a1', 'b1', 'c1']
    ]
    inner2 = [
        ['a2', 'b2', 'c2']
    ]
    inner3 = [
        ['a3', 'b3', 'c3']
    ]
    inner4 = [
        ['a4', 'b4', 'c4']
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
    
    axs1 = subfigs[0].subplot_mosaic(inner1, sharey = True)
    axs2 = subfigs[1].subplot_mosaic(inner2, sharey = True)
    axs3 = subfigs[2].subplot_mosaic(inner3, sharey = True)
    axs4 = subfigs[3].subplot_mosaic(inner4, sharey = True)

    axs = [axs1, axs2, axs3, axs4]


    # axs = fig.subplot_mosaic(outer)

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    plot_col(1, PATH_GENERAL, dates_lead[0], 'a', axs, [1, 2, 3, 4], [weights[0], 'persistence', 'osisaf', 'freedrift', 'nextsim', 'barents'])

    plot_col(2, PATH_GENERAL, dates_lead[1], 'b', axs, [1, 2, 3, 4], [weights[1], 'persistence', 'osisaf', 'freedrift', 'nextsim', 'barents'])

    plot_col(3, PATH_GENERAL, dates_lead[2], 'c', axs, [1, 2, 3, 4], [weights[2], 'persistence', 'osisaf', 'freedrift', 'nextsim', 'barents']) 



    subfigs[0].suptitle(r'10% concentration contour')
    subfigs[1].suptitle(r'40% concentration contour')
    subfigs[2].suptitle(r'70% concentration contour')
    subfigs[3].suptitle(r'90% concentration contour')

    axs1["a1"].set_title(r'1 day lead time')
    axs1["b1"].set_title(r'2 day lead time')
    axs1["c1"].set_title(r'3 day lead time')

    axs4['a4'].legend(loc = 'outside lower left')


    # fig.subplots_adjust(right = 0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    mapper = ScalarMappable(norm = Normalize(3.5, 20), cmap = cm.thermal)

    # cb = fig.colorbar(mapper, ax = axs5['a5'], orientation = 'vertical', fraction = .05, pad = -.9, extend = 'max', ticks = [5, 10, 15, 20], format = mticker.FixedFormatter(['5', '10', '15', '> 20']))
    # cb.outline.set_color('k')

    fig.supylabel('Ice edge displacement error [km]')

    fig.savefig(f"{figname}", dpi = 300)

    exit()

    dummy_fig = plt.figure()
    dummy_subs = dummy_fig.subfigures(nrows = 3)

    fig = plt.figure(figsize = (24, 15), constrained_layout = True)
    subfigs = fig.subfigures(nrows = 1)

    inner1 = [
        ['a1', 'b1', 'c1']
    ]
    inner2 = [
        ['a2', 'b2', 'c2']
    ]
    inner3 = [
        ['a3', 'b3', 'c3']
    ]
    inner4 = [
        ['a4', 'b4', 'c4']
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
    
    axs1 = subfigs.subplot_mosaic(inner1, sharey = True)
    axs2 = dummy_subs[0].subplot_mosaic(inner2, sharey = True)
    axs3 = dummy_subs[1].subplot_mosaic(inner3, sharey = True)
    axs4 = dummy_subs[2].subplot_mosaic(inner4, sharey = True)

    axs = [axs1, axs2, axs3, axs4]


    # axs = fig.subplot_mosaic(outer)

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    plot_col(1, PATH_GENERAL, dates_lead[0], 'a', axs, [1, 2, 3, 4], [weights[0], 'persistence', 'osisaf', 'freedrift', 'nextsim', 'barents'])

    plot_col(2, PATH_GENERAL, dates_lead[1], 'b', axs, [1, 2, 3, 4], [weights[1], 'persistence', 'osisaf', 'freedrift',  'nextsim', 'barents'])

    plot_col(3, PATH_GENERAL, dates_lead[2], 'c', axs, [1, 2, 3, 4], [weights[2], 'persistence', 'osisaf', 'freedrift',  'nextsim', 'barents']) 



    subfigs.suptitle(r'10% concentration contour')
    # subfigs[1].suptitle(r'40% concentration contour')
    # subfigs[2].suptitle(r'70% concentration contour')
    # subfigs[3].suptitle(r'90% concentration contour')

    axs1["a1"].set_title(r'1 day lead time')
    axs1["b1"].set_title(r'2 day lead time')
    axs1["c1"].set_title(r'3 day lead time')

    axs1['a1'].legend()

    for i in ['a', 'b', 'c']:
        axs1[f"{i}1"].set_xticks([1,2,3,4])
        axs1[f"{i}1"].set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])


    # fig.subplots_adjust(right = 0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    # cb = fig.colorbar(mapper, ax = axs5['a5'], orientation = 'vertical', fraction = .05, pad = -.9, extend = 'max', ticks = [5, 10, 15, 20], format = mticker.FixedFormatter(['5', '10', '15', '> 20']))
    # cb.outline.set_color('k')

    fig.supylabel('Ice edge displacement error [km]')

    fig.savefig(f"alt_version_test.pdf", dpi = 300)


if __name__ == "__main__":
    main()
