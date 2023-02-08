#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-3
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeMetricsPersistance.py $SGE_TASK_ID