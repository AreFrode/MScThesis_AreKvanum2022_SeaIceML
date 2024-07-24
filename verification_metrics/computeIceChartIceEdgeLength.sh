#!/bin/bash -f
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=8G,mem_free=8G,h_data=8G 
#$ -q research-r8.q
#$ -M arefk@met.no
#$ -m bae
#$ -o /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/OUT/OUT_$JOB_NAME.$JOB_ID
#$ -e /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/ERR/OUT_$JOB_NAME.$JOB_ID
## ---------------------------

echo "Got $NSLOTS slots."

module use /modules/MET/rhel8/user-modules
module load Python/3.10.4

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/verification_metrics/computeIceChartIceEdgeLength.py