import numpy as np
import h5py


file_new = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast/2021/08/PreparedSample_20210802.hdf5"
file_old = "/lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/two_day_forecast_bak/2021/08/PreparedSample_20210802.hdf5"
#

with h5py.File(file_new) as new:
    xwind_new = np.stack((new['ts0/xwind'][:], new['ts1/xwind'][:]))
    ywind_new = np.stack((new['ts0/ywind'][:], new['ts1/ywind'][:]))

with h5py.File(file_old) as old:
    xwind_old = np.stack((old['ts0/xwind'][:], old['ts1/xwind'][:]))
    ywind_old = np.stack((old['ts0/ywind'][:], old['ts1/ywind'][:]))

#
for lt in range(0, 2):
	#
	diff_wind_x = xwind_new[lt] - xwind_old[lt]
	diff_wind_y = ywind_new[lt] - ywind_old[lt]
	#
	print("Lead time:" + str(lt) + " Max difference wind x-y", np.nanmax(diff_wind_x), np.nanmax(diff_wind_y), np.nanpercentile(diff_wind_x, 99), np.nanpercentile(diff_wind_y, 99))

