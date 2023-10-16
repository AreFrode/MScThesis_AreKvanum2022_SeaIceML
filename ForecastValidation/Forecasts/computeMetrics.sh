#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

weights="weights_09031047"

# module use module use /modules/MET/centos7/GeneralModules

# module load Python-devel/3.8.7

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate /lustre/storeB/users/arefk/conda_envs/conda_compute

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeMetrics.py $weights
 