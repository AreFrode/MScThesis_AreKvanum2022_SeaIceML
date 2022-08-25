# Current setup of hdf5 patched data

The files are ordered by month, as there were 36 months of data, one thread per month which each patched and generated one file. Thus, a single file only contains valid days for the given month. For the current setup, a valid day requires EITHER that an AROME_Arctic file or that an IceChart is produced for that current date. Thus, weekends are omitted as an immediate consequence. Furthermore, weekdays are omitted given that an AROME_arctic output was missing for the given day.


*Example file with descriptive group structure and dims*
/sic-threshold/yyyymmdd/field
    [3]        [n.o.     [3 (time series), n.o. patches, 250, 250]
                valid 
                days]

How to optimally permute the data?

My first estimate is to allocate a permutation-matrix, serving as indices for the files. However, the structure makes it a bity tricky since the files are structured by months

Would it be possible to collect the files in one single file? 