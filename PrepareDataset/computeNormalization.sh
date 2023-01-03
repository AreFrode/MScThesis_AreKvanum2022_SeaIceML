#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q bigmem-r8.q
#$ -l h_rss=200G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/logs

module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

# python code.py lead_time osisaf_trend
python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/computeNormalization.py 2 5
