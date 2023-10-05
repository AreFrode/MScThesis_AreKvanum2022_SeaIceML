import glob
import os

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import pandas as pd

from matplotlib import pyplot as plt, transforms as mtransforms
import datetime

import seaborn as sns

def main():
    # Define paths
    # PATH_FORECAST_STATISTICS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"

    PATH_FORECAST_STATISTICS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/"
    # PATH_FIGURES = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/weights_21021550/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)
    
    # Read available statistics files

    fnames = ('weights_*', 'persistence')
    files = []
    for name in fnames:
        files.extend(glob.glob(f"{PATH_FORECAST_STATISTICS}{name}.csv"))


    files = sorted(files)
    b = [0, 20]
    forecasts = [files[i] for i in b]

    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2021 substitutes D2020
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()


    target = pd.read_csv(files[0], index_col = 0)
    target.index = pd.to_datetime(target.index, format = "%Y-%m-%d")


    for i, idx in zip(range(len(months) - 1), meteorological_seasons):
        target.loc[(target.index >= months[i]) & (target.index < months[i+1]), 'met_index'] = seasonal_names[idx]


    
    target['forecast_name'] = 'persistence'

    columns = 'NIIEE_\d|forecast_name|met_index'

    fetched_forecasts = [target.loc[:, target.columns.str.contains(columns)]]

    for forecast in forecasts[1:]:
        local_filename = filename_regex.findall(forecast)[0]

        local_figure_path = f"{PATH_FIGURES}{local_filename}/"
        if not os.path.exists(local_figure_path):
            os.makedirs(local_figure_path)

        df = pd.read_csv(forecast, index_col = 0)
        df.index = pd.to_datetime(df.index, format = "%Y-%m-%d")


        df['forecast_name'] = local_filename



        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]


        fetched_forecasts.append(df.loc[:, df.columns.str.contains(columns)])
    fetched_dataframe = pd.concat(fetched_forecasts)

    sns.set_theme()
    sns.set(font_scale = 1.1)


    fig, axs = plt.subplot_mosaic('''
                                  abc
                                  def
                                  ''',
                                  figsize = (12, 10))
    
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    contours = ['Open water (<10%)', 'Very open drift ice (10 - 30%)', 'Open drift ice (40 - 60%)', 'Close drift ice (70 - 80%)', 'Very close drift ice (90 - 100%)', 'Fast ice (100%)']
    for i, lab, cont in zip(range(1, 7), labels, contours):

        sns.violinplot(data = fetched_dataframe, x='met_index', y = f'NIIEE_{i}', hue = 'forecast_name', ax = axs[lab], inner = 'quartile', split = True)
        axs[lab].text(0.0, 1.0, f"{lab})", transform=axs[lab].transAxes + trans,
            fontsize='medium', va='bottom')
        
        axs[lab].set(xlabel = None)
        axs[lab].set(ylabel = None)
        axs[lab].set_title(cont)
        
        if lab != 'b':
            axs[lab].legend_.remove()

        else:
            axs[lab].legend_.set_title('Forecast name')
            for t, l in zip(axs[lab].legend_.texts, ['Persistence', 'Deep learning']):
                t.set_text(l)

        if i < 4:
            axs[lab].set(xticklabels = [])
    
    axs['a'].set_ylim(bottom = 0)
    axs['f'].set_ylim(bottom = 0)
    fig.suptitle('Seasonal distribution of average ice edge displacement (NIIEE)')
    fig.supylabel('NIIEE [km]')
    # fig.supxlabel('Season')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{PATH_FIGURES}niiee.pdf")

if __name__ == "__main__":
    main()
