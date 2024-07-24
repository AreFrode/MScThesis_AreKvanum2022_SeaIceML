#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=8G,mem_free=8G
#$ -t 1-12
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/logs/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate development-11-2023

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/fetchAMSR2trend.py $SGE_TASK_ID 