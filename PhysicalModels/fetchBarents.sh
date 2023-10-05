#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -t 1-12
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/logs/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate /lustre/storeB/users/arefk/conda_envs/conda_compute

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/fetchBarents.py $SGE_TASK_ID nextsim
python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/fetchBarents.py $SGE_TASK_ID amsr2