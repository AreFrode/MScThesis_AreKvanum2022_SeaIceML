#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/logs/

# 'barents' / 'ml' / 'nextsim' / 'osisaf' / 'persistence'
forecast="ml"

# 1 / 2 / 3
lead_time="2"

# 'nextsim' / 'amsr2'
target_grid="nextsim"

# If not forecast='ml', nothing happens
weights="weights_09031802"


module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/commonData.py $target_grid
python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/computeMetrics.py $forecast $lead_time $target_grid $weights