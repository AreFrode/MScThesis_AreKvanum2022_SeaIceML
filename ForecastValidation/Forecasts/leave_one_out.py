import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt, ticker as mticker


def loader(fname, lead):
    return pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{lead}/weights_{fname}.csv", usecols=['date', 'NIIEE_2'], index_col = 0).mean()


def main():
    
    no_sic = []
    no_trend = []
    no_lsmask = []
    no_t2m = []
    no_xwind = []
    no_ywind = []
    no_winds = []
    no_arome = []

    persistence = []
    baseline = []

    weights = ["weights_08031256", "weights_21021550", "weights_09031047"]


    no_sic.append(loader('11111137', 1))
    no_sic.append(loader('07031244', 2))
    no_sic.append(loader('12110754', 3))

    no_trend.append(loader('11110727', 1))
    no_trend.append(loader('06031648', 2))
    no_trend.append(loader('12110548', 3))

    no_lsmask.append(loader('10112157', 1))
    no_lsmask.append(loader('06031337', 2))
    no_lsmask.append(loader('12110345', 3))

    no_t2m.append(loader('09111244', 1))
    no_t2m.append(loader('09031802', 2))
    no_t2m.append(loader('11111757', 3))

    no_xwind.append(loader('09111858', 1))
    no_xwind.append(loader('05031824', 2))
    no_xwind.append(loader('11112011', 3))

    no_ywind.append(loader('10110752', 1))
    no_ywind.append(loader('05031508', 2))
    no_ywind.append(loader('11112218', 3))

    no_winds.append(loader('10111141', 1))
    no_winds.append(loader('03031044', 2))
    no_winds.append(loader('12110156', 3))

    no_arome.append(loader('10111436', 1))
    no_arome.append(loader('09031539', 2))
    no_arome.append(loader('12110023', 3))

    for t, weight in enumerate(weights, 1):
        persistence.append(pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/persistence.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean())
        baseline.append(pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/{weights[t-1]}.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean())
    NIIEEs = []
    NIIEEs.append([no_t2m[0], no_arome[0], no_winds[0], no_ywind[0], no_xwind[0], no_lsmask[0], no_trend[0], no_sic[0]])
    NIIEEs.append([no_t2m[1], no_arome[1], no_winds[1], no_ywind[1], no_xwind[1], no_lsmask[1], no_trend[1], no_sic[1]])
    NIIEEs.append([no_t2m[2], no_arome[2], no_winds[2], no_ywind[2], no_xwind[2], no_lsmask[2], no_trend[2], no_sic[2]])

    no_t2m = np.array(no_t2m)
    no_sic = np.array(no_sic)
    no_trend = np.array(no_trend)
    no_lsmask = np.array(no_lsmask)
    no_winds = np.array(no_winds)
    no_xwind = np.array(no_xwind)
    no_ywind = np.array(no_ywind)
    no_arome = np.array(no_arome)
    baseline = np.array(baseline)
    peristence = np.array(persistence)

    print(no_sic - baseline)
    print((no_sic - baseline).mean())
    print(f"{(no_arome - baseline)=}")

    print((no_winds - baseline).mean())
    print((no_t2m - baseline).mean())

    print((no_trend - baseline).mean())

    tick_labels = ['t2m', 'arome', 'winds', 'ywind', 'xwind', 'lsmask', 'trend', 'sic']


    sns.set_theme(context = 'poster')
    sns.set_palette('colorblind')


    fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (14, 11.5), constrained_layout = True)

    for i in range(3):
        axs[i].plot(NIIEEs[i], '--o')
        axs[i].hlines(baseline[i], color = 'k', ls = 'dashed', xmin = 0, xmax = 7, linewidth = .5, label='Deep learning (all predictors)')
        axs[i].hlines(persistence[i], color = 'red', ls = 'dashed', xmin = 0, xmax = 7, linewidth = .5, label = 'Persistence')
        axs[i].grid(axis = 'x')
        axs[i].set_xticks([])
        axs[i].set_title(f"{i+1}-day lead time")

    axs[2].yaxis.set_minor_locator(mticker.AutoMinorLocator())
    axs[2].set_xticks(range(8), tick_labels)

    # ax.set_ylim(top = 5)

    axs[0].legend(framealpha = 1)

    fig.supxlabel('Removed predictor')
    fig.supylabel('Ice edge displacement error [km]')
    fig.savefig('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/figures/leaveoneout.png', dpi = 300)
    


if __name__ == "__main__":
    main()
