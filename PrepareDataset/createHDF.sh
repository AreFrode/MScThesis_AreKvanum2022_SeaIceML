#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=16G,mem_free=16G,h_data=16G
#$ -t 1-48
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate development-11-2023


# 'None'/'open_ocean'/'reduced_classes'
MODE='amsr2_input'

# python createHDF.py lead_time osisaf_trend member
python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/createHDF.py $SGE_TASK_ID $MODE

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
