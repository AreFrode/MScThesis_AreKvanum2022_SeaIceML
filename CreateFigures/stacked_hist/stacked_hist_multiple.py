import sys
sys.path.append("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from WMOcolors import cm

def month_sorter(column):
    """Sort function"""
    sorter = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Des']
    correspondence = {sort: order for order, sort in enumerate(sorter)}
    return column.map(correspondence)

def order_sort(column):
    """Sort function"""
    sorter = ['A', 'B', 'C', 'D']
    correspondence = {sort: order for order, sort in enumerate(sorter)}
    return column.map(correspondence)

def season_sort(column):
    sorter = ['DJF', 'MAM', 'JJA', 'SON']
    correspondence = {sort: order for order, sort in enumerate(sorter)}
    return column.map(correspondence)


def read_data(path):
    df = pd.read_csv(path, index_col = 0)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')

    months = df.resample('M').mean()

    data = []

    for i in range(len(months)):
        data.append([])
        current_month = months.iloc[i]
        month_total = current_month.sum()
        for j in range(len(current_month)):
            data[i].append(current_month.iloc[j] / month_total)

    df_frac = pd.DataFrame(data, columns = ['0', '1', '2', '3', '4', '5', '6'], index = months.index)
    df_frac = df_frac.set_index(np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    return df_frac

def read_data_seasons(path):
    df = pd.read_csv(path, index_col = 'met_index')
    df = df.fillna(0)

    seasons = df.groupby(by = 'met_index').mean(numeric_only = True)

    data = []

    for i in range(len(seasons)):
        data.append([])
        current_season = seasons.iloc[i]
        season_total = current_season.sum()
        for j in range(len(current_season)):
            data[i].append(current_season.iloc[j] / season_total)

    df_frac = pd.DataFrame(data, columns = ['IFOW', 'OW', 'VODI', 'ODI', 'CDI', 'VCDI', 'FI'], index = seasons.index)
    df_frac = df_frac.sort_index(key = season_sort)

    return df_frac


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe,
    Inspired by (https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars)"""

    sns.set_context('poster')
    sns.set_theme()

    categories = ['Ice Free Open Water', 'Open Water', 'Very Open Drift Ice', 'Open Drift Ice', 'Close Drift Ice', 'Very Close Drift Ice', 'Fast Ice']


    figsize = (20, 14)

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    axe = fig.add_subplot()

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      # linewidth=0,
                      edgecolor = 'k',
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      fontsize=20,
                      **kwargs)  # make bar plots

    
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 2 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_xlabel('')
    axe.set_ylabel('Fraction of ice categories across domain', fontsize = 20)
    

    # Add invisible data to add another legend
    
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))


    leg_h = h[:n_col]
    l1 = axe.legend(leg_h[::-1], categories[::-1], fontsize = 28, loc = 'lower right', framealpha = 1, title = 'Ice categories', title_fontsize = 28)
    if labels is not None:
        l2 = plt.legend(n, labels, loc = 'lower right', bbox_to_anchor = (1, .35), fontsize = 28, framealpha = 1, title = 'Product', title_fontsize = 28)

    axe.add_artist(l1)
    
    axe.set_xlim([-0.3, 3.65])

    return axe



def main():
    path_ic = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_sic_counts_seasons.csv'
    path_lead1 = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_lead1_counts_amsr2_seasons.csv'
    path_lead2 = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_lead2_counts_amsr2_seasons.csv'
    path_lead3 = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_lead3_counts_amsr2_seasons.csv'
    path_amsr2 = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/CreateFigures/stacked_hist/2022_amsr2_counts_seasons.csv'

    
    wmo = [np.array([255., 255., 255.])/255., np.array([150., 200., 255.])/255., np.array([140., 255., 160.])/255., np.array([255., 255., 0.])/255., np.array([255., 125., 7.])/255., np.array([255., 0., 0.])/255., np.array([150., 150., 150.])/255.]

    

    df_ic = read_data_seasons(path_ic)

    df_1 = read_data_seasons(path_lead1)

    df_2 = read_data_seasons(path_lead2)

    df_3 = read_data_seasons(path_lead3)

    df_amsr2 = read_data_seasons(path_amsr2)

    # print(f"{df_amsr2['IFOW'].mean()=}")
    # print(f"{df_amsr2['OW'].mean()=}")

    # print(f"{df_amsr2['VCDI'].mean()=}")
    # print(f"{df_amsr2['FI'].mean()=}")
    


    # print(f"{df_ic['IFOW'].mean()=}")
    # print(f"{df_ic['OW'].mean()=}")

    # print(f"{df_ic['VCDI'].mean()=}")
    # print(f"{df_ic['FI'].mean()=}")
    # df_all.plot(kind = 'bar', stacked = True, rot = 0, fontsize = 20, ax = ax,color = wmo, edgecolor = 'k')
    # df_ic.plot(kind = 'bar', stacked = True, rot = 0, fontsize = 20, ax = ax,color = wmo, edgecolor = 'k')
    # df_1.plot(kind = 'bar', stacked = True, rot = 0, fontsize = 20, ax = ax, color = wmo, edgecolor = 'k')
    # df_2.plot(kind = 'bar', stacked = True, rot = 0, fontsize = 20, ax = ax, color = wmo, edgecolor = 'k')
    # df_3.plot(kind = 'bar', stacked = True, rot = 0, fontsize = 20, ax = ax, color = wmo, edgecolor = 'k')


    '''
    axs['a'].set_title('Sea ice charts', fontsize = 18)
    axs['b'].set_title('Deep learning 1-day lead time', fontsize = 18)
    axs['c'].set_title('Deep learning 2-day lead time', fontsize = 18)
    axs['d'].set_title('Deep learning 3-day lead time', fontsize = 18)

    handles, labels = axs['a'].get_legend_handles_labels()
    axs['a'].legend(handles[::-1], categories[::-1], fontsize = 18)
    axs['b'].get_legend().remove()
    axs['c'].get_legend().remove()
    axs['d'].get_legend().remove()


    fig.supxlabel('Month', fontsize = 18)
    fig.supylabel('Fraction of total sea ice cover', fontsize = 18)
    fig.suptitle('2022 Monthly sea ice category fraction', fontsize = 18)
    '''

    plot_clustered_stacked([df_amsr2, df_ic, df_1, df_2, df_3], ["AMSR2", "IceCharts", "1-day lead time", "2-day lead time", "3-day lead time"], cmap=cm.sea_ice_chart())
    # plot_clustered_stacked([df_ic, df_1, df_2, df_3], ["IceCharts", "1-day lead time", "2-day lead time", "3-day lead time"], cmap=cm.sea_ice_chart())

    plt.savefig('2022-siccategory_histogram_seasons_amamsr2inputt.pdf',bbox_inches = 'tight', dpi = 300)

    # print(df_ic)
    # print(df_1)
    # print(df_2)
    # print(df_3)
    

if __name__ == "__main__":
    main()
