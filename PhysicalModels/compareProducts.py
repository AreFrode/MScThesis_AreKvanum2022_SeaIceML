import glob
import os
import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts")

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import pandas as pd

from matplotlib import pyplot as plt
from compareForecasts import IceEdgeStatisticsFigure

import seaborn as sns


def main():
    sns.set_theme()
    sns.despine()

    # Define paths
    PATH_NEXTSIM = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_2/nextsim.csv"
    PATH_OSISAF = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_2/osisaf_trend_5/osisaf.csv"
    PATH_ML = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_2/osisaf_trend_5/ml.csv"
    PATH_BARENTS = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_2/barents.csv"
    PATH_PERSISTENCE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/lead_time_2/persistence.csv"
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"
    PATH_FIGURES = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/figures/"

    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    # Read Climatological iceedge
    climatological_ice_edge = pd.read_csv(PATH_CLIMATOLOGICAL_ICEEDGE, index_col = 0)

    # Comment this out if leap year
    climatological_ice_edge = climatological_ice_edge.drop('02-29')

    # Update to 2022
    # climatological_ice_edge.index = '2021-' + climatological_ice_edge.index
    climatological_ice_edge.index = '2022-' + climatological_ice_edge.index
    climatological_ice_edge.set_index(pd.DatetimeIndex(pd.to_datetime(climatological_ice_edge.index)))
    
    # Convert length from [m] to [km]
    climatological_ice_edge['ice_edge_length'] = climatological_ice_edge['ice_edge_length'] / 1000.
    
    # Read available statistics files

    # files = (PATH_NEXTSIM, PATH_OSISAF, PATH_ML, PATH_BARENTS, PATH_PERSISTENCE)
    files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)

    # set common dates from lowest denominator
    dates = pd.read_csv(PATH_ML)['date'].array


    # Define met seasons

    meteorological_seasons = [0,0,1,1,1,2,2,2,3,3,3,0] # D2022 substitutes D2021
    seasonal_names = ['DJF', 'MAM', 'JJA', 'SON']

    months = pd.date_range('2022-01-01','2023-01-01', freq='MS').strftime("%Y-%m-%d").tolist()


    # Create figure classes

    IIEE_figure = IceEdgeStatisticsFigure(f"{PATH_FIGURES}Custom_noOSi.png", "Normalized IIEE comparison (IceEdge >= 10%)", "IIEE [km]")


    # Define list where forecasts are appended
    fetched_forecasts = [] 

    for forecast in files:
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['climatological_length'] = climatological_ice_edge['ice_edge_length']
        df['Normalized_IIEE'] = df['IIEE_2'] / df['climatological_length']

        df['forecast_name'] = local_filename

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        if local_filename == 'barents':
            df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]

        fetched_forecasts.append(df[['Normalized_IIEE', 'forecast_name', 'met_index']])
    

    fetched_dataframe = pd.concat(fetched_forecasts)

    IIEE_figure.setYLimit(125)
    IIEE_figure.addBoxPlot(df = fetched_dataframe, x_label = 'met_index', y_label = 'Normalized_IIEE', hue_label = 'forecast_name')
    IIEE_figure.savefig('Months', 'Normalized_IIEE [km]')
    


if __name__ == "__main__":
    main()