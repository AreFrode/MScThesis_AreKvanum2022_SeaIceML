#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-10
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

weights="weights_21021550"
field="ywind"

if (($SGE_TASK_ID < 10))   
then
    slice="-55"
else
    slice="-56"
fi

module use module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeSwappedMetrics.py $weights $field $slice $SGE_TASK_ID
