#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/logs/

module use /modules/MET/centos7/GeneralModules

module load Python-devel/3.8.7

weights1='weights_30010909'
weights2='weights_05011118'
weights3='weights_30011419'

for i in {1..6..1}
do
    for j in 'nextsim' 'amsr2'
    do
        python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/compareProducts.py 1 $j $weights1 $i
        python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/compareProducts.py 2 $j $weights2 $i
        python3 /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/PhysicalModels/compareProducts.py 3 $j $weights3 $i
    done
done