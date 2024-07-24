import calendar
import pandas as pd


def load_climatological_ice_edge(year, concentration, lead_time, product):
    """Loads the climatological ice edge as a dataframe

    Args:
        year (int): Which year to load for, determines mainly leap year
        lead_time (int): Adjusts the date to the forecast bulletin date

    Returns:
        DataFrame: Lead time adjusted dataframe
    """
    
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"

    if product == 'nextsim':
        PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/icechart_ice_edge.csv"
    # Read Climatological iceedge
    # climatological_ice_edge = pd.read_csv(PATH_CLIMATOLOGICAL_ICEEDGE, index_col = 0, usecols=[concentration])

    elif product == 'amsr2':
        PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/amsr2_ice_edge.csv"

    else:
        exit('No valid product identifier-string was passed')

    climatological_ice_edge = pd.read_csv(PATH_CLIMATOLOGICAL_ICEEDGE, index_col = 0, usecols = ['date',concentration])

    # Comment this out if leap year
    if not (calendar.isleap(year)):
        try:
            climatological_ice_edge = climatological_ice_edge.drop('02-29')

        except KeyError:
            pass

    # Update to 2022
    # climatological_ice_edge.index = '2021-' + climatological_ice_edge.index
    climatological_ice_edge.index = f'{year}-' + climatological_ice_edge.index

    # Ice edge valid for bulletin date not valid date
    climatological_ice_edge.index = pd.to_datetime(climatological_ice_edge.index, format="%Y-%m-%d") - pd.DateOffset(lead_time)
    
    climatological_ice_edge.index = climatological_ice_edge.index.map(lambda x : x.replace(year = year))
    climatological_ice_edge = climatological_ice_edge.sort_index()

    return climatological_ice_edge
