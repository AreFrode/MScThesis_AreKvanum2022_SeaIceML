#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-24
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module use /modules/MET/centos7/GeneralModules/
module load Python-devel/3.8.7

# 'None'/'open_ocean'/'reduced_classes'
MODE='None'

# python createHDF.py lead_time osisaf_trend member
python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/createHDF.py $SGE_TASK_ID $MODE

# remove problematic date 20210223
if [ $SGE_TASK_ID -eq 62 ]
then
    if [ -z "$MODE" ]
    then
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_1/2021/02/PreparedSample_v20210224_b20210223.hdf5
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/2021/02/PreparedSample_v20210225_b20210223.hdf5
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_3/2021/02/PreparedSample_v20210226_b20210223.hdf5
    else
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"$MODE"/lead_time_1/2021/02/PreparedSample_v20210224_b20210223.hdf5
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"$MODE"/lead_time_2/2021/02/PreparedSample_v20210225_b20210223.hdf5
        rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/"$MODE"/lead_time_3/2021/02/PreparedSample_v20210226_b20210223.hdf5
    fi
fi
