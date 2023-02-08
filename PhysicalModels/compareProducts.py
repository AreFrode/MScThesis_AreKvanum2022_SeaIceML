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

    lead_time = sys.argv[1]
    grid = sys.argv[2]
    weights = sys.argv[3]
    contour = sys.argv[4]

    # Define paths
    PATH_NEXTSIM = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/nextsim.csv"
    PATH_OSISAF = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/osisaf.csv"
    PATH_ML = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/{weights}.csv"
    PATH_BARENTS = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/barents.csv"
    PATH_PERSISTENCE = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/{grid}_grid/lead_time_{lead_time}/persistence.csv"
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"
    PATH_FIGURES = f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/figures/lead_time_{lead_time}/contour_{contour}/"

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
    # files = (PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS)
    # files = [PATH_NEXTSIM, PATH_ML, PATH_BARENTS]
    files = [PATH_NEXTSIM, PATH_PERSISTENCE, PATH_ML, PATH_BARENTS, PATH_OSISAF]

    # set common dates

    dates = pd.concat([pd.read_csv(file, index_col = 0) for file in files], axis=1, join = 'inner').index.array


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

    percentages = ['>=0', '>=10', '>=40', '>=70', '>=90', '=100']

    IIEE_figure = IceEdgeStatisticsFigure(figname, f"Normalized IIEE comparison (IceEdge {percentages[int(contour)-1]}%)", "IIEE [km]")


    # Define list where forecasts are appended
    fetched_forecasts = [] 

    for forecast in files:
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['climatological_length'] = climatological_ice_edge['ice_edge_length']
        df['Normalized_IIEE'] = df[f'IIEE_{int(contour)}'] / df['climatological_length']

        df['forecast_name'] = local_filename

        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        # if local_filename == 'barents':
        #     df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]

        fetched_forecasts.append(df[['Normalized_IIEE', 'forecast_name', 'met_index']])
    

    fetched_dataframe = pd.concat(fetched_forecasts)

    IIEE_figure.setYLimit(125)
    IIEE_figure.addBoxPlot(df = fetched_dataframe, x_label = 'met_index', y_label = 'Normalized_IIEE', hue_label = 'forecast_name')
    IIEE_figure.savefig('Months', 'Normalized_IIEE [km]')
    


if __name__ == "__main__":
    main()