import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

def main():
    PATH_GENERAL = '/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/Data/amsr2_grid/'
    PATH_FIGURES = f"{PATH_GENERAL[:-11]}figures/thesis_figs/"
    columns = ['NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']
    common_title = 'concentration contour'
    subtitles = [rf'10% {common_title}', rf'40% {common_title}', rf'70% {common_title}', rf'90% {common_title}']

    ml = []
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/weights_08031256.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/weights_21021550.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/weights_09031047.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    ml_grid = np.zeros((4, 3))

    pers = []
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/amsr2_persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/amsr2_persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/amsr2_persistence.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    pers_grid = np.zeros_like(ml_grid)

    # Confusing namecheme, remember amsr2 linear trend
    osi = []
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/amsr2_trend.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/amsr2_trend.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/amsr2_trend.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    osi_grid = np.zeros_like(ml_grid)

    freedrift = []
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/freedriftAMSR2.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/freedriftAMSR2.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/freedriftAMSR2.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    freedrift_grid = np.zeros_like(ml_grid)

    nextsim = []
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_1/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_2/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim.append(pd.read_csv(f"{PATH_GENERAL}lead_time_3/nextsim.csv", index_col = 0, usecols = ['date', 'NIIEE_2', 'NIIEE_3', 'NIIEE_4', 'NIIEE_5']))
    nextsim_grid = np.zeros_like(ml_grid)

    tick_labels = ['1-day', '2-day', '3-day']

    common_dates = [pd.concat([ml[i], pers[i], osi[i], nextsim[i]], axis=1, join = 'inner').index.array for i in range(3)]

    for k in range(4):
        for i in range(3):
            ml_grid[k,i] = ml[i][columns[k]][ml[i][columns[k]].index.isin(common_dates[i])].mean()
            pers_grid[k,i] = pers[i][columns[k]][pers[i][columns[k]].index.isin(common_dates[i])].mean()
            osi_grid[k,i] = osi[i][columns[k]][osi[i][columns[k]].index.isin(common_dates[i])].mean()
            freedrift_grid[k,i] = freedrift[i][columns[k]][freedrift[i][columns[k]].index.isin(common_dates[i])].mean()
            nextsim_grid[k,i] = nextsim[i][columns[k]][nextsim[i][columns[k]].index.isin(common_dates[i])].mean()
    

    cont0 = np.mean(nextsim_grid[0,:] - ml_grid[0,:])
    cont1 = np.mean(nextsim_grid[1,:] - ml_grid[1,:])
    cont2 = np.mean(nextsim_grid[2,:] - ml_grid[2,:])
    cont3 = np.mean(nextsim_grid[3,:] - ml_grid[3,:])


    print(f"{ml_grid[0]=}")
    print(f"{np.diff(ml_grid[0])=}")

    print(f"{pers_grid[0]=}")
    print(f"{np.diff(pers_grid)=}")
    
    print(f"{osi_grid[0]=}")
    print(f"{np.diff(osi_grid)=}")

    print(f"{freedrift_grid[0]=}")
    print(f"{np.diff(freedrift_grid)=}")
    
    print(f"{nextsim_grid[0]=}")
    print(f"{np.diff(nextsim_grid)=}")


    
    # print(np.mean([cont1 - cont0, cont2 - cont1, cont3 - cont2]))
    # exit()

    figname = f"{PATH_FIGURES}model_intercomparisson_leadtime_amsr2.pdf"

    mosaic_labels = ['a', 'b', 'c', 'd']

    sns.set_theme(context = 'poster')
    sns.set_palette('deep')

    fig = plt.figure(figsize = (14, 11.5), constrained_layout = True)
    axs = fig.subplot_mosaic(
        [
        ['a', 'b'],
        ['c', 'd']
        ]
    )

    for i, lab in enumerate(mosaic_labels):
        axs[lab].plot(ml_grid[i], '-o', label = 'Deep learning')
        axs[lab].plot(pers_grid[i], '-o', label = 'AMSR2 persistence')
        axs[lab].plot(osi_grid[i], '-o', label = 'AMSR2 linear trend')
        axs[lab].plot(freedrift_grid[i], '-o', label = 'AMSR2 freedrift')
        axs[lab].plot(nextsim_grid[i], '-o', label = 'NeXtSIM')
        axs[lab].set_title(f"{subtitles[i]}")
        axs[lab].set_xticks(np.arange(3), tick_labels)
        axs[lab].grid(axis = 'x')

    axs['a'].legend()
    axs['a'].set_xticks([])
    axs['b'].set_xticks([])
    fig.suptitle('Mean annual forecast error, AMSR2 target')
    fig.supylabel('Ice edge displacement error [km]')
    fig.supxlabel('Lead time')
    fig.savefig(f"{figname}", dpi = 300)


if __name__ == "__main__":
    main()
