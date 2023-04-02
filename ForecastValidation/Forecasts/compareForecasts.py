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
    def __init__(self, fname, title, ylabel, figsize):
        self.figure = plt.figure(figsize=figsize)
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
        
    def addViolinPlot(self, df, x_label, y_label, hue_label):
        # df.boxplot(by = 'met_index', column='IIEE', ax = self.ax)
        sns.violinplot(data = df, 
                    x = x_label, 
                    y = y_label, 
                    hue = hue_label,
                    palette='deep',
                    inner = 'box',
                    ax = self.ax)
        

    def addNewErrorPlot(self, df, label):
        self.ax.errorbar(x= df.index, y = df['mean'], yerr = (df['min'], df['max']), label = label)

    def moveLegend(self, pos):
        sns.move_legend(self.ax, pos, bbox_to_anchor=(1,1))

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

    # Create figure classes

    # ice_edge_length_figure = IceEdgeStatisticsFigure(f"{PATH_FIGURES}ice_edge_length_seasons_box.png", "Ice Edge Length Comparison", "Ice Edge Length [km]")
    IIEE_figure = IceEdgeStatisticsFigure(f"{PATH_FIGURES}IIEE_spread_no_t2m.png", "Normalized IIEE comparison", "IIEE [km]", (10,10))


    # Fill figures

    target = pd.read_csv(files[0], index_col = 0, usecols=['date', 'forecast_length', 'target_length', 'NIIEE_2'])
    target.index = pd.to_datetime(target.index, format = "%Y-%m-%d")


    # target['target_length'] = target['target_length'] / 1000.
    # target['forecast_length'] = target['forecast_length'] / 1000.

    target_monthly_mean = target.groupby(pd.PeriodIndex(target.index, freq="M")).mean()

    for i, idx in zip(range(len(months) - 1), meteorological_seasons):
        target.loc[(target.index >= months[i]) & (target.index < months[i+1]), 'met_index'] = seasonal_names[idx]

    target_seasonal = target.groupby('met_index')
    
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'target_length', 'target')
    # ice_edge_length_figure.addBoxPlot(target, 'met_index', 'forecast_length', 'persistance')
    
    target['forecast_name'] = 'persistence'

    # monthly_mean_lengths = []
    # monthly_mean_IIEE = []

    fetched_forecasts = [target[['NIIEE_2', 'forecast_name', 'met_index']]] 

    seasonal_mean_lengths = []
    seasonal_mean_IIEE = []

    # b = [4, 5, -4, -2] Old forecast index

    b = [11, 15]
    forecasts = [files[i] for i in b]

    for forecast in forecasts:
        local_filename = filename_regex.findall(forecast)[0]

        local_figure_path = f"{PATH_FIGURES}{local_filename}/"
        if not os.path.exists(local_figure_path):
            os.makedirs(local_figure_path)

        df = pd.read_csv(forecast, index_col = 0)
        df.index = pd.to_datetime(df.index, format = "%Y-%m-%d")

        # df['forecast_length'] = df['forecast_length'] / 1000.

        df['forecast_name'] = local_filename


        monthly_mean = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean()

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        
        
        met_seasons = df.groupby('met_index').mean()

        # plot_area(monthly_mean, local_figure_path)
        # seasonal_mean_lengths.append((met_seasons['forecast_length']).to_frame().rename(columns={'forecast_length': local_filename}))
        seasonal_mean_IIEE.append((met_seasons['NIIEE_2']).to_frame())

        # df['days_beat_persistance'] = df['Normalized_IIEE'] < target['Normalized_IIEE']

        
        # days_beat_persistance_months = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).sum()
        # samples_each_month = days_beat_persistance.groupby(pd.PeriodIndex(days_beat_persistance.index, freq="M")).size()

        # ratio = monthly_mean['Normalized_IIEE'] / target_monthly_mean['Normalized_IIEE']

        # print(ratio)
        # print(ratio.mean())

        # ice_edge_length_figure.addNewPlot(met_seasons['forecast_length'], local_filename)
        # IIEE_figure.addNewPlot(met_seasons['Normalized_IIEE'], local_filename)
        
        # print(f"model: {local_filename}, beat IIEE from persistance {days_beat_persistance.sum()}/{days_beat_persistance.size} days")

        # for i, beat, total in zip(range(1,13), days_beat_persistance_months, samples_each_month):
            # pass
            # print(f"month {i}: {beat}/{total}")

        fetched_forecasts.append(df[['NIIEE_2', 'forecast_name', 'met_index']])

    
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
    IIEE_figure.addViolinPlot(df = fetched_dataframe, x_label = 'met_index', y_label = 'NIIEE_2', hue_label = 'forecast_name')
    IIEE_figure.savefig('Months', 'Normalized_IIEE [km]')
    


if __name__ == "__main__":
    main()
