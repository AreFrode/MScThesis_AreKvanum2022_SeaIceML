import glob
import os

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import datetime

import seaborn as sns

class IceEdgeStatisticsFigure:
    def __init__(self, fname, title, ylabel):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.fname = fname
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)

    def setYLimit(self, upper):
        self.ax.set_ylim(top = upper)
    
    def addPersistence(self, df):
        df.plot(ax = self.ax, label = 'Persistence', ls='--', ms='5')

    def addNewPlot(self, df, label):
        df.plot(ax = self.ax, label = label)

    def addBoxPlot(self, df, x_label, y_label, hue_label):
        # df.boxplot(by = 'met_index', column='IIEE', ax = self.ax)
        sns.boxplot(data = df, 
                    x = x_label, 
                    y = y_label, 
                    hue = hue_label,
                    palette='deep',
                    showmeans = True,
                    meanprops={"marker": 'D', "markeredgecolor": 'black',
                      "markerfacecolor": 'firebrick'},
                    whis = [5,95],
                    ax = self.ax)

    def addNewErrorPlot(self, df, label):
        self.ax.errorbar(x= df.index, y = df['mean'], yerr = (df['min'], df['max']), label = label)

    def savefig(self, xlabel, ylabel):
        # self.ax.set_xlim([datetime.date(2020, 12, 30), datetime.date(2022, 1, 2)])
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        plt.legend()
        self.figure.savefig(self.fname)


def plot_area(df, path, num_classes=7):
    for i in range(num_classes):
        ax_area = df[f'target_area{i}'].plot(label='target')
        df[f'forecast_area{i}'].plot(ax=ax_area, label='forecast')
        ax_area.set_title(f'SIC area for concenration class {i}')
        ax_area.set_ylabel('Ice Edge contour area [km^2]')
        plt.legend()
        plt.savefig(f"{path}class{i}.png")
        ax_area.clear()

def main():
    sns.set_theme()
    sns.despine()

    # Define paths
    # PATH_FORECAST_STATISTICS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"
    PATH_PERSISTENCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/persistance.csv"
    PATH_FORECAST_STATISTICS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/"
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"
    PATH_FIGURES = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_2/figures/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    # Read available statistics files

    fnames = ('weights_*', 'persistence')
    files = []
    for name in fnames:
        files.extend(glob.glob(f"{PATH_FORECAST_STATISTICS}{name}.csv"))


    files = sorted(files)


    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2021 substitutes D2020
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    # Update to 2022
    # months = pd.date_range('2021-01-01','2022-01-01', freq='MS').strftime("%Y-%m-%d").tolist()
    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()

    # set common dates
    # dates = pd.concat(files, axis = 1, join = 'inner').index.array


    target = pd.read_csv(files[0], index_col = 0, usecols=['date', 'forecast_length', 'target_length', 'NIIEE_2'])
    target.index = pd.to_datetime(target.index, format = "%Y-%m-%d")


    for i, idx in zip(range(len(months) - 1), meteorological_seasons):
        target.loc[(target.index >= months[i]) & (target.index < months[i+1]), 'met_index'] = seasonal_names[idx]

    
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'target_length', 'target')
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'forecast_length', 'persistance')
    
    target['forecast_name'] = 'persistence'


    for forecast in files[1:]:
        local_filename = filename_regex.findall(forecast)[0]

        local_figure_path = f"{PATH_FIGURES}{local_filename}/"
        if not os.path.exists(local_figure_path):
            os.makedirs(local_figure_path)

        df = pd.read_csv(forecast, index_col = 0)
        df.index = pd.to_datetime(df.index, format = "%Y-%m-%d")


        df['forecast_name'] = local_filename

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        df['days_beat_persistence'] = df['NIIEE_2'] < target['NIIEE_2']

        
        # days_beat_persistance_months = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).sum()
        # samples_each_month = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).size()

        # ratio = monthly_mean['Normalized_IIEE'] / target_monthly_mean['Normalized_IIEE']

        # print(ratio)
        # print(ratio.mean())

        # ice_edge_length_figure.addNewPlot(met_seasons['forecast_length'], local_filename)
        # IIEE_figure.addNewPlot(met_seasons['Normalized_IIEE'], local_filename)
        
        print(f"model: {local_filename}, beat NIIEE from persistence {df['days_beat_persistence'].sum()}/{df['days_beat_persistence'].size} days")


        # df_group = df.groupby(['met_index'])['days_beat_persistence'].sum()
        df_group = df.groupby(pd.Grouper(freq='M'))['days_beat_persistence'].sum()
        # df_size = df.groupby(['met_index']).size()
        df_size = df.groupby(pd.Grouper(freq='M')).size()

        df_days = df_group / df_size

        print(df_days)

        plt.figure()

        ax = df_days.plot()
        
        ax.set_ylim(bottom = 0.4, top = 1.1)
        ax.set_ylabel('Percentage of days with NIIEE < Persistence')
        
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(['', '50%', '60%', '70%', '80%', '90%', '100%', ''])
        

        plt.savefig(f"{PATH_FIGURES}{local_filename}_NIIEE_beat_persistence.png")

        # for i, beat, total in zip(range(1,13), days_beat_persistance_months, samples_each_month):
            # pass
            # print(f"month {i}: {beat}/{total}")



    
    




if __name__ == "__main__":
    main()
