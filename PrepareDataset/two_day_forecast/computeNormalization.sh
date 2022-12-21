#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q bigmem-r8.q
#$ -l h_rss=200G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/two_day_forecast/logs

module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/two_day_forecast/computeNormalization.py
