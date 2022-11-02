#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q research-el7.q
#$ -l h_vmem=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/logs/visualizations

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

conda activate production-03-2022

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/visualize_prediction.py