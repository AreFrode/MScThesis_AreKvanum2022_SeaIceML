import glob
import os
import sys
sys.path.append("/home/arefk/Documents/Lustre/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts")

import re
FILENAME_PATTERN = r"(?<=\/)(\w+)(?=\.)"
filename_regex = re.compile(FILENAME_PATTERN)

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from compareForecasts import IceEdgeStatisticsFigure

import seaborn as sns


def main():
    sns.set_theme(context = "talk")
    sns.despine()

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

    fnames = ['NeXtSIM', 'Persistence', 'Deep learning', 'OSI SAF trend', 'Barents-2.5']

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

    figname = f"/home/arefk/uio/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/poster_figs/NIIEE_mean.svg"

    percentages = ['>=0', '>=10', '>=40', '>=70', '>=90', '=100']
    figsize = (11,11)

    IIEE_figure = IceEdgeStatisticsFigure(figname, f"Model intercomparisson (mean of [10-40], [40-70], [70-90], [90-100] contours)", "IIEE [km]", figsize)


    # Define list where forecasts are appended
    fetched_forecasts = [] 

    for forecast, name in zip(files, fnames):
        local_filename = filename_regex.findall(forecast)[0]

        df = pd.read_csv(forecast, index_col = 0)
        
        # Find common dates
        df = df[df.index.isin(dates)]

        df['forecast_name'] = name

        df['Normalized_IIEE'] = df.filter(regex=r'NIIEE_?[2-5]').mean(axis=1)
            
        for i, idx in zip(range(len(months) - 1), meteorological_seasons):
            df.loc[(df.index >= months[i]) & (df.index < months[i+1]), 'met_index'] = seasonal_names[idx]

        if local_filename == 'barents':
            df = df[(df.met_index != 'DJF') & (df.met_index != 'MAM')]

        fetched_forecasts.append(df[['Normalized_IIEE', 'forecast_name', 'met_index']])
    

    fetched_dataframe = pd.concat(fetched_forecasts)

    IIEE_figure.setYLimit(125)
    IIEE_figure.addBoxPlot(df = fetched_dataframe, x_label = 'met_index', y_label = 'Normalized_IIEE', hue_label = 'forecast_name')

    # IIEE_figure.moveLegend("lower left")

    IIEE_figure.savefig('Seasons', 'Ice edge displacement error [km] (Normalized IIEE)')
    


if __name__ == "__main__":
    main()