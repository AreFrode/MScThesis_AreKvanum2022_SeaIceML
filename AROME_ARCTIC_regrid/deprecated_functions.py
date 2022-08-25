def load_TCC(id: str, dataset: Dataset, x, y, time, output_dataset: Dataset, name: str, unit: str, standard_name: str, nx: int = 3220 - 526, ny: int = 2979 - 194) -> Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]:
	"""loads a 4d variable, e.g. arome_arctic_full_* ([time, height, x, y])

	Args:
		id (str): Name of variable in arome arctic
		dataset (Dataset): dataset where the variable is loaded from
		output_dataset (Dataset): output netcdf file
		name (str): name of variable in output
		unit (str): unit of variable in output
		standard_name (str): descriptive name in output
		nx (int, optional): length of target grid in x direction. Defaults to 3220-526.
		ny (int, optional): length of target grid in y direction. Defaults to 2979-194.
		slicer (Slice, optional): height values to obtain. Defaults to 0.

	Returns:
		Tuple[np.ndarray, np.ndarray, netCDF4._netCDF4.Variable]: output array for storing computed values, arome values and output netCDF4 variable
	"""

	id_ICgrid = np.zeros((3, ny, nx))
	id_input = np.zeros((len(time), len(y), len(x)))

	# There has to be a better way than this, slow loop many FLOPs
	for i in range(len(time)):
		id_input[i,:,:] = dataset.variables[id][i,...].sum(axis=0)

	id_arome = np.pad(id_input, ((0,0), (1,1), (1, 1)), 'constant', constant_values=np.nan)
	id_out = output_dataset.createVariable(name, 'd', ('time', 'y', 'x'))
	id_out.units = unit
	id_out.standard_name = standard_name

	return (id_ICgrid, id_arome, id_out)