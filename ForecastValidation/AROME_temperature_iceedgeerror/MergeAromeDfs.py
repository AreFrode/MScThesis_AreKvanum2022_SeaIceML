import sys
import os
import glob

import numpy as np
import pandas as pd

def main():
    paths = sorted(glob.glob(f"/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/AROME_temperature_iceedgeerror/2022/**/*.csv"))

    df = pd.concat((pd.read_csv(f, index_col=0) for f in paths))

    df.to_csv("/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/AROME_temperature_iceedgeerror/merged_df.csv")



if __name__ == "__main__":
    main()