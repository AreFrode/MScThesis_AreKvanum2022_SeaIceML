#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -q research-r8.q
#$ -l h_rss=20G
#$ -l mem_free=20G
#$ -t 1-10
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/logs/predictions

#Old centos7 definitions

#module load Python-devel/3.8.7-TF2.4.1

#python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/predict_validation.py weights_27111420

#New Singularity definitions on rhel8

weights="weights_21021550"

module use /modules/MET/rhel8/user-modules/

module load go/1.19.1
module load singularity/3.10.2

singularity exec -B /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML:/mnt $HOME/TFcontainer/tensorflow_latest-gpu.sif python /mnt/SimpleUNET/RunModel/predict_swapped_validation.py $weights $SGE_TASK_ID