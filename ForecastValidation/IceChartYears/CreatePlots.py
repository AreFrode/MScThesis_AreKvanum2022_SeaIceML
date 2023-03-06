import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def main():
    df = pd.read_csv("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/IceChartYears/merged_df.csv", index_col = 0)

    df.index = pd.to_datetime(df.index)

    df['doy'] = df.index.dayofyear
    df['year'] = df.index.year

    piv = pd.pivot_table(df, index = ['doy'], columns = ['year'], values = ['extent'])

    sns.set_theme()

    plt.figure()

    ax = piv.plot()
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Sea ice extent [contour >= 10% - 40%]')

    plt.savefig("Persistence_SIE.png")

    piv2 = pd.pivot_table(df, index = ['doy'], columns = ['year'], values = ['niiee'])

    plt.figure()

    ax = piv2.plot(marker = '.', ls ='none')
    ax.set_xlabel('Day of year')
    ax.set_ylabel('NIIEE [contour >= 10% - 40%]')

    plt.savefig("Persistence_NIIEE.png")

if __name__ == "__main__":
    main()