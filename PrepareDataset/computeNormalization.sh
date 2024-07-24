#$ -S /bin/bash
#$ -l h_rt=04:00:00
#$ -q research-r8.q
#$ -l h_rss=100G,mem_free=100G,h_data=100G
#$ -t 1-3
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/logs

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate development-11-2023

# python code.py lead_time osisaf_trend
python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/computeNormalization.py $SGE_TASK_ID
