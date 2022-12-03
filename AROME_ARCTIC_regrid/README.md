# AROME_ARCTIC_REGRID

This folder contains code for generating .netcdf samples of select AROME Arctic (AA) data reprojected from a 2.5km onto a 1km grid using Nearest Neighbor interpolation.

The directory contains 5 sub-directories. 

1. **data_processing_files** contain output log files from the GridEngine submission scripts, divided into std.out, std.err and program files used for parallel execution.
2. **OLD_IceChartGrid** contain old code from Spring22 where AA was projected onto the grid of the Ice Charts, this has since been deprecated.
3. **one_day_forecast** Deprecated code used to create AA samples which can be used for constructing training-ready samples for a model predicting with a one day lead time. *NOTE* that this script will have to be updated to be used with current code implementations
4. **testingdata** directory used to dump script files executed while developing/debugging
5. **two_day_forecast** contains two files. The first file is a Python-script which regrids AA onto a 1km spatial resolution. Currently the fields x, y, lat, lon, sst, t2m, xwind and ywind are extracted. The AA forecast intitiated at 18 is fetched, and the variables are stored as two "daily" means in the resulting netcdf file with the following timestep structure (0-18)(18-42) (in layman's terms (day 0) 18:00 - 12:00 (day 1), (day 1) 12:00-12:00 (day 2)). Note that the Sea Ice concentration forecast is initiated at (day 0) 15:00 and target at (day 2) 15:00.