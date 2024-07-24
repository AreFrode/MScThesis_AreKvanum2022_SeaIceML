#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G,mem_free=8G,h_data=8G
#$ -t 1-9
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

weights=(
    weights_03031044
    weights_09031802
    weights_13031648
	weights_05031508
	weights_05031824
	weights_06031337
	weights_06031648
	weights_07031244
	weights_09031539
)

# module use module use /modules/MET/centos7/GeneralModules

# module load Python-devel/3.8.7

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate development-11-2023

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeMetrics.py ${weights[$SGE_TASK_ID - 1]}
