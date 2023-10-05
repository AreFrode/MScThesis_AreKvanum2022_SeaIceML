import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt, dates as mdates, ticker as mticker


    
def main():
    PATH_CLIMATOLOGICAL_ICEEDGE = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/Data/climatological_ice_edge.csv"

    # Read Climatological iceedge
    climatological_ice_edge = pd.read_csv(PATH_CLIMATOLOGICAL_ICEEDGE, index_col = 0)

    # Comment this out if leap year
    climatological_ice_edge = climatological_ice_edge.drop('02-29')

    # Update to 2022
    # climatological_ice_edge.index = '2021-' + climatological_ice_edge.index
    climatological_ice_edge.index = '2022-' + climatological_ice_edge.index
    climatological_ice_edge = climatological_ice_edge.set_index(pd.DatetimeIndex(pd.to_datetime(climatological_ice_edge.index)))
    
    # Plotting

    sns.set_theme()
    fig, ax = plt.subplots(figsize = (12,8))
    # ax = climatological_ice_edge['ice_edge_length'].plot()
    sns.lineplot(data = climatological_ice_edge, x = climatological_ice_edge.index, y = '15%', ax = ax)
    ax.tick_params(labelsize = 14)
    ax.set_ylim(bottom = 0, top = 4000)

    # ax.xaxis.set_major_locator(mticker.LinearLocator(numticks = 12))
    months = mdates.MonthLocator()
    date_form = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)


    ax.set_xlabel('Dates', fontsize = 16)
    ax.set_ylabel('Ice Edge Length [km]', fontsize = 16)
    ax.set_title('Climatological sea ice edge length', fontsize = 16)

    fig.savefig('clim_iceedge.pdf')


if __name__ == "__main__":
    main()