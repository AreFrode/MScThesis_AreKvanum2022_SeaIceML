#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-3
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate development-11-2023

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeMetricsPersistance.py $SGE_TASK_ID