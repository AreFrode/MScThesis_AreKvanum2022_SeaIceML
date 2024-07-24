import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

def main():
    PATH_GENERAL = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/nextsim_grid/'
    PATH_FIGURES = f"{PATH_GENERAL[:-13]}figures/thesis_figs/"
    common_title = 'concentration contour'
    columns = ['NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']
    subtitles = [rf'10% {common_title}', rf'40% {common_title}', rf'70% {common_title}', rf'90% {common_title}']

    ml = []
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_08031256.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_21021550.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_09031047.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml_grid = np.zeros((4, 3))

    ml2 = []
    ml2.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_13051126.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml2.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_13051605.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml2.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_13051918.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml2_grid = np.zeros((4, 3))

    pers = []
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers_grid = np.zeros((4, 3))

    tick_labels = ['1-day', '2-day', '3-day']

    common_dates = [pd.concat([ml[i], pers[i]], axis=1, join = 'inner').index.array for i in range(3)]

    for k in range(4):
        for i in range(3):
            ml_grid[k,i] = ml[i][columns[k]][ml[i][columns[k]].index.isin(common_dates[i])].mean()
            ml2_grid[k,i] = ml2[i][columns[k]][ml2[i][columns[k]].index.isin(common_dates[i])].mean()
            pers_grid[k,i] = pers[i][columns[k]][pers[i][columns[k]].index.isin(common_dates[i])].mean()
    
    ml = np.array(ml_grid[0]).ravel()
    ml2 = np.array(ml2_grid[0]).ravel()
    pers = np.array(pers_grid[0]).ravel()
    
    print(f"{ml=}")
    print(f"{np.diff(ml)=}")

    print(f"{ml2=}")
    print(f"{np.diff(ml2)=}")

    print(f"{pers=}")
    print(f"{np.diff(pers)=}")
    

    figname = f"{PATH_FIGURES}model_intercomparisson_leadtime_nextsim_mlappended.pdf"

    sns.set_theme(context = 'poster')
    sns.set_palette(palette = 'deep')

    fig = plt.figure(figsize = (14, 11.5), constrained_layout = True)
    ax = fig.add_subplot()

 
    ax.plot(ml, '-o', label = 'Deep learning')
    ax.plot(ml2, '-o', label = '12T AROME Arctic appended')
    ax.plot(pers, '-o', label = 'Persistence')

    ax.set_xticks(np.arange(3), tick_labels)
    ax.set_title('Mean annual forecast error')
    ax.set_ylabel('Ice edge displacement error [km]')
    ax.set_xlabel('Lead time')
    ax.grid(axis = 'x')

    ax.legend()
    fig.savefig(f"{figname}", dpi = 300)


    figname2 = f"{PATH_FIGURES}model_intercomparisson_leadtime_nextsim_all_mlappended.png"

    fig = plt.figure(figsize = (14, 11.5), constrained_layout = True)
    axs = fig.subplot_mosaic(
        [
        ['a', 'b'],
        ['c', 'd']
        ]
    )

    mosaic_labels = ['a', 'b', 'c', 'd']

    for i, lab in enumerate(mosaic_labels):
        axs[lab].plot(ml_grid[i], '-o', label = 'Deep learning')
        axs[lab].plot(ml2_grid[i], '-o', label = '12T AROME Arctic appended')
        axs[lab].plot(pers_grid[i], '-o', label = 'Persistence')
        axs[lab].set_title(f"{subtitles[i]}")
        axs[lab].set_xticks(np.arange(3), tick_labels)
        axs[lab].grid(axis = 'x')

    legend = axs['d'].legend(bbox_to_anchor=[0.9, 0.45])
    axs['a'].set_xticks([])
    axs['b'].set_xticks([])
    fig.suptitle('Mean annual forecast error')
    fig.supylabel('Ice edge displacement error [km]')
    fig.supxlabel('Lead time')
    fig.savefig(f"{figname2}", dpi = 300)


if __name__ == "__main__":
    main()

