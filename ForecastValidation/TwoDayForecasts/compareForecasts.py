import glob
import os

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import pandas as pd

from matplotlib import pyplot as plt
import datetime

import seaborn as sns

class IceEdgeStatisticsFigure:
    def __init__(self, fname, title, ylabel):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.fname = fname
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
    
    def addPersistantce(self, df):
        df.plot(ax = self.ax, label = 'Persistance', ls='--', ms='5')

    def addNewPlot(self, df, label):
        df.plot(ax = self.ax, label = label)

    def addBoxPlot(self, df, x_label, y_label, hue_label):
        # df.boxplot(by = 'met_index', column='IIEE', ax = self.ax)
        sns.boxplot(data = df, 
                    x = x_label, 
                    y = y_label, 
                    hue = hue_label, 
                    flierprops={"marker": "x"},
                    showmeans = True,
                    meanprops={"marker": 'D', "markeredgecolor": 'black',
                      "markerfacecolor": 'firebrick'},
                    whis = [5,95],
                    ax = self.ax)

    def addNewErrorPlot(self, df, label):
        self.ax.errorbar(x= df.index, y = df['mean'], yerr = (df['min'], df['max']), label = label)

    def savefig(self):
        # self.ax.set_xlim([datetime.date(2020, 12, 30), datetime.date(2022, 1, 2)])
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
    # Define paths
    PATH_FORECAST_STATISTICS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/Data/"
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"
    PATH_FIGURES = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/figures/"

    # Read Climatological iceedge
    climatological_ice_edge = pd.read_csv(PATH_CLIMATOLOGICAL_ICEEDGE, index_col = 0)

    # Comment this out if leap year
    climatological_ice_edge = climatological_ice_edge.drop('02-29')

    climatological_ice_edge.index = '2021-' + climatological_ice_edge.index
    climatological_ice_edge.set_index(pd.DatetimeIndex(pd.to_datetime(climatological_ice_edge.index)))
    
    # Convert length from [m] to [km]
    climatological_ice_edge['ice_edge_length'] = climatological_ice_edge['ice_edge_length'] / 1000.


    # Read available statistics files

    fnames = ('weights_*', 'persistance')
    files = []
    for name in fnames:
        files.extend(glob.glob(f"{PATH_FORECAST_STATISTICS}{name}.csv"))
    files = sorted(files)


    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2021 substitutes D2020
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2021-01-01','2022-01-01', freq='MS').strftime("%Y-%m-%d").tolist()


    # Create figure classes

    # ice_edge_length_figure = IceEdgeStatisticsFigure(f"{PATH_FIGURES}ice_edge_length_seasons_box.png", "Ice Edge Length Comparison", "Ice Edge Length [km]")
    IIEE_figure = IceEdgeStatisticsFigure(f"{PATH_FIGURES}IIEE_spread_seasons.png", "Normalized IIEE comparison", "IIEE [km]")


    # Fill figures

    target = pd.read_csv(files[0], index_col = 0, usecols=['date', 'forecast_length', 'target_length', 'IIEE'])
    target['Normalized_IIEE'] = target['IIEE'] / climatological_ice_edge['ice_edge_length']
    target['target_length'] = target['target_length'] / 1000.
    target['forecast_length'] = target['forecast_length'] / 1000.

    target_monthly_mean = target.groupby(pd.PeriodIndex(target.index, freq="M")).mean()

    for i, idx in zip(range(len(months) - 1), meteorological_seasons):
        target.loc[(target.index > months[i]) & (target.index < months[i+1]), 'met_index'] = seasonal_names[idx]

    target_seasonal = target.groupby('met_index')
    
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'target_length', 'target')
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'forecast_length', 'persistance')
    
    target['forecast_name'] = 'persistance'

    # monthly_mean_lengths = []
    # monthly_mean_IIEE = []

    fetched_forecasts = [target[['Normalized_IIEE', 'forecast_name', 'met_index']]] 

    seasonal_mean_lengths = []
    seasonal_mean_IIEE = []

    b = [4, -3, -1]

    for forecast in [files[i] for i in b]:
        local_filename = filename_regex.findall(forecast)[0]

        local_figure_path = f"{PATH_FIGURES}{local_filename}/"
        if not os.path.exists(local_figure_path):
            os.makedirs(local_figure_path)

        df = pd.read_csv(forecast, index_col = 0)
        df['climatological_length'] = climatological_ice_edge['ice_edge_length']
        df['Normalized_IIEE'] = df['IIEE'] / df['climatological_length']
        df['forecast_length'] = df['forecast_length'] / 1000.

        df['forecast_name'] = local_filename

        monthly_mean = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean()

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index > months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        fetched_forecasts.append(df[['Normalized_IIEE', 'forecast_name', 'met_index']])
        met_seasons = df.groupby('met_index').mean()

        # plot_area(monthly_mean, local_figure_path)
        seasonal_mean_lengths.append((met_seasons['forecast_length']).to_frame().rename(columns={'forecast_length': local_filename}))
        seasonal_mean_IIEE.append((met_seasons['IIEE']).to_frame().rename(columns={'IIEE': local_filename}))

        days_beat_persistance = df['Normalized_IIEE'] < target['Normalized_IIEE']

        days_beat_persistance_months = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).sum()
        samples_each_month = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).size()

        ratio = monthly_mean['Normalized_IIEE'] / target_monthly_mean['Normalized_IIEE']

        # print(ratio)
        # print(ratio.mean())

        # ice_edge_length_figure.addNewPlot(met_seasons['forecast_length'], local_filename)
        # IIEE_figure.addNewPlot(met_seasons['Normalized_IIEE'], local_filename)
        
        # print(f"model: {local_filename}, beat IIEE from persistance {days_beat_persistance.sum()}/{days_beat_persistance.size} days")

        for i, beat, total in zip(range(1,13), days_beat_persistance_months, samples_each_month):
            pass
            # print(f"month {i}: {beat}/{total}")

    
    # seasonal_mean_lengths_df = pd.concat(seasonal_mean_lengths, axis = 1)

    # seasonal_mean_lengths_df['mean'] = seasonal_mean_lengths_df.mean(axis = 1)
    # monthly_mean_lengths_df['std'] = monthly_mean_lengths_df.std(axis=1)

    # seasonal_mean_lengths_df['min'] = seasonal_mean_lengths_df.min(axis = 1)
    # seasonal_mean_lengths_df['max'] = seasonal_mean_lengths_df.max(axis = 1)

    # ice_edge_length_figure.addNewErrorPlot(seasonal_mean_lengths_df, 'forecasts')

    # ice_edge_length_figure.savefig()

    fetched_dataframe = pd.concat(fetched_forecasts)

    # IIEE_figure.addPersistantce(target_seasonal_mean['Normalized_IIEE'])

    # seasonal_mean_IIEE_df = pd.concat(seasonal_mean_IIEE, axis = 1)

    # seasonal_mean_IIEE_df['mean'] = seasonal_mean_IIEE_df.mean(axis = 1)
    # monthly_mean_IIEE_df['std'] = monthly_mean_IIEE_df.std(axis=1)

    # seasonal_mean_IIEE_df['min'] = seasonal_mean_IIEE_df.min(axis = 1)
    # seasonal_mean_IIEE_df['max'] = seasonal_mean_IIEE_df.max(axis = 1)
    # IIEE_figure.addNewErrorPlot(monthly_mean_IIEE_df, 'forecasts')
    IIEE_figure.addBoxPlot(df = fetched_dataframe, x_label = 'met_index', y_label = 'Normalized_IIEE', hue_label = 'forecast_name')
    IIEE_figure.savefig()
    


if __name__ == "__main__":
    main()