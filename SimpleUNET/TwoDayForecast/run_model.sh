#$ -S /bin/bash
#$ -q gpu-r8.q
#$ -l h=gpu-05.ppi.met.no
#$ -l h_rt=24:00:00
#$ -l h_rss=16G
#$ -l mem_free=4G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/logs/qsub_model_run_logs

#Old centos7 definitions

#source /modules/rhel8/conda/install/etc/profile.d/conda.sh

#conda activate TensorFlowGPU-03-2022

#CUDA_VISIBLE_DEVICES=0 python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/run_model.py > /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/logs/$(date -d "today" +"%d%m%H%M").log 2> /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/errors/$(date -d "today" +"%d%m%H%M").log

#New Singularity definitions running on rhel8

COMMAND='python -u /mnt/SimpleUNET/TwoDayForecast/run_model.py'

module use /modules/MET/rhel8/user-modules

module load go/1.19.1
module load singularity/3.10.2

singularity exec -B /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML:/mnt --nv $HOME/TFcontainer/tensorflow_latest-gpu.sif sh -c "$COMMAND > /mnt/SimpleUNET/TwoDayForecast/logs/$(date -d "today" +"%d%m%H%M").log 2> /mnt/SimpleUNET/TwoDayForecast/errors/$(date -d "today" +"%d%m%H%M").log"