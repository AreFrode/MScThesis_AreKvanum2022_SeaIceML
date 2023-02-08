#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-el7.q
#$ -l h_vmem=10G
#$ -t 1-3
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/data_processing_files/OUT/

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-03-2022

python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/validateHDF.py $SGE_TASK_ID