import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt, transforms as mtransforms


def main():
    df = pd.read_csv("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/IceChartYears/merged_df.csv", index_col = 0)

    df.index = pd.to_datetime(df.index)

    df['doy'] = df.index.dayofyear
    df['year'] = df.index.year

    sns.set_theme()
    sns.color_palette('deep', as_cmap = True)
    sns.set(font_scale = 1.3)

    # plt.rcParams.update({'font.size': 22})

    fig = plt.figure(figsize = (7.5, 10))
    axs = fig.subplots(nrows = 2, ncols = 1)

    sns.lineplot(data = df, x = 'doy', y = 'extent', hue = 'year', palette = 'deep', ax = axs[0])

    sns.scatterplot(data = df, x = 'doy', y = 'niiee', hue = 'year', palette = 'deep', ax = axs[1])

    axs[0].set_xlabel('')
    axs[0].set_ylabel('Sea ice extent [km^2]')
    axs[0].legend_.remove()

    axs[1].set_xlabel('Day of year')
    axs[1].set_ylabel('Sea ice edge displacement error [km]')

    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    labels = ['a)', 'b)']
    for ax, label in zip(axs, labels):
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom')


    # ax = piv.plot()
    # ax.set_xlabel('Day of year')
    # ax.set_ylabel('Sea ice extent [contour >= 10%]')

    # plt.savefig("Persistence_SIE.png")

    # plt.figure()

    # ax = piv2.plot(marker = '.', ls ='none')
    # ax.set_xlabel('Day of year')
    # ax.set_ylabel('NIIEE [contour >= 10%]')
    fig.suptitle('Sea Ice Chart sea ice extent and NIIEE \n with persistence for the 10% contour')

    fig.savefig("Persistence_NIIEE_SIE.png")

if __name__ == "__main__":
    main()