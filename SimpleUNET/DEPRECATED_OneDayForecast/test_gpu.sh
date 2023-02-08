#!/bin/bash -f
#$ -N IsGPUavail
#$ -l h_rt=1:00:00
#$ -l h_vmem=4500M
#$ -S /bin/bash
#$ -q gpu.q
#$ -M arefk@met.no
#$ -m bae
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/data_processing_files/OUT/OUT_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/data_processing_files/ERR/ERR_$JOB_NAME.$JOB_ID.$HOSTNAME.$TASK_ID
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/data_processing_files/OUT/

echo "Got $NSLOTS slots"

module load Python-devel/3.8.7-TF2.4.1

python3 test_gpu.py