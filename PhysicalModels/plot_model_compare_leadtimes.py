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
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_08031256.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_21021550.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_09031047.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_31052147.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_01060203.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_01060501.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_13051126.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_13051605.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    # ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_13051918.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml_grid = np.zeros((4, 3))

    pers = []
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers_grid = np.zeros((4, 3))

    osi = []
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/osisaf.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/osisaf.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/osisaf.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi_grid = np.zeros((4, 3))

    freedrift = []
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/freedrift.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/freedrift.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/freedrift.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift_grid = np.zeros((4, 3)) 

    nextsim = []
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim_grid = np.zeros((4, 3))

    tick_labels = ['1-day', '2-day', '3-day']

    common_dates = [pd.concat([ml[i], pers[i], osi[i], nextsim[i]], axis=1, join = 'inner').index.array for i in range(3)]

    for k in range(4):
        for i in range(3):
            ml_grid[k,i] = ml[i][columns[k]][ml[i][columns[k]].index.isin(common_dates[i])].mean()
            pers_grid[k,i] = pers[i][columns[k]][pers[i][columns[k]].index.isin(common_dates[i])].mean()
            osi_grid[k,i] = osi[i][columns[k]][osi[i][columns[k]].index.isin(common_dates[i])].mean()
            freedrift_grid[k,i] = freedrift[i][columns[k]][freedrift[i][columns[k]].index.isin(common_dates[i])].mean()
            nextsim_grid[k,i] = nextsim[i][columns[k]][nextsim[i][columns[k]].index.isin(common_dates[i])].mean()
    
    ml = np.array(ml_grid[0]).ravel()
    pers = np.array(pers_grid[0]).ravel()
    osi = np.array(osi_grid[0]).ravel()
    freedrift = np.array(freedrift_grid[0]).ravel()
    nextsim = np.array(nextsim_grid[0]).ravel()
    
    print(f"{ml=}")
    print(f"{np.diff(ml)=}")

    print(f"{pers=}")
    print(f"{np.diff(pers)=}")
    
    print(f"{osi=}")
    print(f"{np.diff(osi)=}")

    print(f"{freedrift=}")
    print(f"{np.diff(freedrift)=}")
    
    print(f"{nextsim=}")
    print(f"{np.diff(nextsim)=}")

    # print(np.mean(np.array(ml) / np.array(pers)))

    # exit('EARLY QUIT TO GET RESULTS, NO FIGURE SAVED')

    figname = f"{PATH_FIGURES}model_intercomparisson_leadtime_nextsim_amsr2input.pdf"


    sns.set_theme(context = 'poster')
    sns.set_palette(palette = 'deep')

    fig = plt.figure(figsize = (14, 11.5), constrained_layout = True)
    ax = fig.add_subplot()

 
    ax.plot(ml, '-o', label = 'Deep learning')
    ax.plot(pers, '-o', label = 'Persistence')
    ax.plot(osi, '-o', label = 'OSI SAF linear trend')
    ax.plot(freedrift, '-o', label = 'Freedrift')
    ax.plot(nextsim, '-o', label = 'NeXtSIM')

    ax.set_xticks(np.arange(3), tick_labels)
    ax.set_title('Mean annual forecast error')
    ax.set_ylabel('Ice edge displacement error [km]')
    ax.set_xlabel('Lead time')
    ax.grid(axis = 'x')

    ax.legend()
    fig.savefig(f"{figname}", dpi = 300)


    figname2 = f"{PATH_FIGURES}model_intercomparisson_leadtime_nextsim_all_amsr2input.pdf"

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
        axs[lab].plot(pers_grid[i], '-o', label = 'Persistence')
        axs[lab].plot(osi_grid[i], '-o', label = 'OSI SAF linear trend')
        axs[lab].plot(freedrift_grid[i], '-o', label = 'Freedrift')
        axs[lab].plot(nextsim_grid[i], '-o', label = 'NeXtSIM')
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

