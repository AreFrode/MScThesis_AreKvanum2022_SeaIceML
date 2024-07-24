#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-10
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/logs

# weights="weights_08031256"
# weights="weights_21021550"
weights="weights_09031047"
fields=(
    lsmask
    sic
    t2m
    trend
    xwind
    ywind
)

if (($SGE_TASK_ID < 10))   
then
    slice=(
        "-56"
        "-53"
        "-53"
        "-55"
        "-55"
        "-55"
    )
else
    slice=(
        "-57"
        "-54"
        "-54"
        "-56"
        "-56"
        "-56"
    )
fi

# module use module use /modules/MET/centos7/GeneralModules
source /modules/rhel8/conda/install/etc/profile.d/conda.sh

# module load Python-devel/3.8.7
conda activate development-11-2023

for i in "${!fields[@]}"; do
    python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/ForecastValidation/Forecasts/computeSwappedMetrics.py $weights ${fields[i]} ${slice[i]} $SGE_TASK_ID
done
