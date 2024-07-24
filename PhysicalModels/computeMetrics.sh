#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=8G,mem_free=8G,h_data=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/logs/

# 'barents' / 'ml' / 'nextsim' / 'osisaf' / 'persistence'
forecast="ml"

# 1 / 2 / 3
lead_time="2"

# 'nextsim' / 'amsr2'
target_grid="nextsim"

# If not forecast='ml', nothing happens
# weights="weights_08031256"
# weights="weights_21021550"
# weights="weights_09031047"

# New def 6 contours
# weights="weights_22051953"
# weights="weights_23050006"
# weights="weights_23050301"

# Appended models
# weights="weights_13051126"
# weights="weights_13051605"
# weights="weights_13051918"

# single output models
# weights="weights_23051706"
# weights="weights_23052053"
weights="weights_01050203"


# module use /modules/MET/centos7/GeneralModules
# module use /modules/MET/rhel8/user-modules/

# module load Python-devel/3.8.7
# module load Python/3.10.4
source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate development-11-2023

python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/commonData.py $target_grid
python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/computeMetrics.py $forecast $lead_time $target_grid $weights