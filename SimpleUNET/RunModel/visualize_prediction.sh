#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/logs/visualizations

weights="weights_21021550"

source /modules/rhel8/conda/install/etc/profile.d/conda.sh

# conda activate production-03-2022

conda activate /lustre/storeB/users/arefk/conda_envs/cartopy_plot

python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/visualize_prediction.py $weights