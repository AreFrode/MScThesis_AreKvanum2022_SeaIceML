#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_vmem=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/logs

module use module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/TwoDayForecasts/computeMetrics.py weights_30111329