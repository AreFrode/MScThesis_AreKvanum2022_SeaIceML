#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -q research-r8.q
#$ -l h_rss=40G
#$ -l mem_free=40G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/Saliency/Contour/logs

module use /modules/MET/rhel8/user-modules/

module load go/1.19.1
module load singularity/3.10.2

singularity exec -B /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML:/mnt $HOME/TFcontainer/tensorflow_latest-gpu.sif python /mnt/Saliency/Contour/CreateContourData.py 
