#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-48
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module use /modules/MET/centos7/GeneralModules/
module load Python-devel/3.8.7

python3 --version

# python createHDF.py lead_time osisaf_trend member
python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/createHDF.py $SGE_TASK_ID

if [ $SGE_TASK_ID -eq 26 ]
then
    rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_1/2021/02/PreparedSample_v20210224_b20210223.hdf5
    rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_2/2021/02/PreparedSample_v20210225_b20210223.hdf5
    rm /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/Data/lead_time_3/2021/02/PreparedSample_v20210226_b20210223.hdf5
fi