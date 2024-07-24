import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt, ticker as mticker


def main():
    weights = ["weights_08031256", "weights_21021550", "weights_09031047"]

    sic_ensembles = []
    trend_ensembles = []
    lsmask_ensembles = []
    t2m_ensembles = []
    xwind_ensembles = []
    ywind_ensembles = []

    persistences = []
    baselines = []

    for t, weight in enumerate(weights, 1):
        sic_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_sic.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        trend_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_trend.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        lsmask_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_lsmask.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        t2m_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_t2m.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        xwind_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_xwind.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        ywind_ensembles.append([pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/swapped/seed_{i}/{weight}_overwritten_ywind.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean() for i in range(1,11)])
        persistences.append(pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/persistence.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean())
        baselines.append(pd.read_csv(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/lead_time_{t}/{weights[t-1]}.csv", usecols = ['date', 'NIIEE_2'], index_col = 0).mean())

    sic_ensembles = np.squeeze(np.array(sic_ensembles))
    trend_ensembles = np.squeeze(np.array(trend_ensembles))
    lsmask_ensembles = np.squeeze(np.array(lsmask_ensembles))
    t2m_ensembles = np.squeeze(np.array(t2m_ensembles))
    xwind_ensembles = np.squeeze(np.array(xwind_ensembles))
    ywind_ensembles = np.squeeze(np.array(ywind_ensembles))

    persistences = np.squeeze(np.array(persistences))
    baselines = np.squeeze(np.array(baselines))

    # print(np.mean(sic_ensembles, axis = 1))
    # print(np.std(sic_ensembles, axis = 1))

    print(xwind_ensembles.mean(axis = 1))
    print(ywind_ensembles.mean(axis = 1))
    print(t2m_ensembles.mean(axis = 1))

    print((trend_ensembles.mean(axis = 1) - baselines).mean())


    print('stds')
    print(f"{sic_ensembles.std(axis = 1)=}")
    print(f"{trend_ensembles.std(axis = 1)=}")
    print(f"{t2m_ensembles.std(axis = 1)=}")
    print(f"{xwind_ensembles.std(axis = 1)=}")
    print(f"{ywind_ensembles.std(axis = 1)=}")


    sns.set_theme(context = 'poster')
    sns.set_palette('deep')

    fig = plt.figure(figsize = (14, 11.5), constrained_layout = True)
    ax = fig.add_subplot()


    ax.errorbar(x = range(3), y = sic_ensembles.mean(axis = 1) - baselines, yerr = sic_ensembles.std(axis = 1), marker = '.', ls = '--', label = 'sic')
    ax.errorbar(x = range(3), y = trend_ensembles.mean(axis = 1) - baselines, yerr = trend_ensembles.std(axis = 1), marker = '.', ls = '--', label = 'trend')
    # ax.errorbar(x = range(3), y = lsmask_ensembles.mean(axis = 1) - baselines, yerr = lsmask_ensembles.std(axis = 1), marker = '.', ls = '--', label = 'lsmask')
    ax.errorbar(x = range(3), y = t2m_ensembles.mean(axis = 1) - baselines, yerr = t2m_ensembles.std(axis = 1), marker = '.', ls = '--', label = 't2m')
    ax.errorbar(x = range(3), y = xwind_ensembles.mean(axis = 1) - baselines, yerr = xwind_ensembles.std(axis = 1), marker = '.', ls = '--', label = 'xwind')
    ax.errorbar(x = range(3), y = ywind_ensembles.mean(axis = 1) - baselines, yerr = ywind_ensembles.std(axis = 1), marker = '.', ls = '--', label = 'ywind')

    # ax.plot(baselines, '.--', label = 'Deep learning', zorder = 10)
    ax.plot(persistences - baselines, '.--', label = 'Persistence', zorder = 10)


    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xticks(range(3), ['1-day', '2-day', '3-day'])

    ax.grid(axis = 'x')
    ax.set_title('Predictor permutation')
    ax.set_ylabel('Ice edge displacement error difference [km]')
    ax.set_xlabel('Lead time')
    ax.legend(loc = 'center left')
    fig.savefig('/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/figures/new_swapped.pdf', dpi = 300)


if __name__ == "__main__":
    main()
