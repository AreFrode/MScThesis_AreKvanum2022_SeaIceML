#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-84
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

module use /modules/MET/centos7/GeneralModules/
module load Python-devel/3.8.7

python3 --version

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/AROME_ARCTIC_regrid/Regrid_AromeArctic_cum.py $SGE_TASK_ID