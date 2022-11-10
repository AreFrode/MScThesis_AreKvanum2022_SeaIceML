#$ -S /bin/bash
#$ -l h_rt=02:00:00
#$ -q research-el7.q
#$ -l h_vmem=200G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/two_day_forecast/logs

module load Python-devel/3.8.7

python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PrepareDataset/two_day_forecast/computeNormalization.py