#$ -S /bin/bash
#$ -q gpu-r8.q
#$ -l h=gpu-05.ppi.met.no
#$ -l h_rt=48:00:00
#$ -l h_rss=16G,mem_free=16G,h_data=16G
#$ -wd /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/RunModel/logs/qsub_model_run_logs

#Old centos7 definitions

#source /modules/rhel8/conda/install/etc/profile.d/conda.sh

#conda activate TensorFlowGPU-03-2022

#CUDA_VISIBLE_DEVICES=0 python /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/run_model.py > /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/logs/$(date -d "today" +"%d%m%H%M").log 2> /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML/SimpleUNET/TwoDayForecast/errors/$(date -d "today" +"%d%m%H%M").log

#New Singularity definitions running on rhel8

module use /modules/MET/rhel8/user-modules

module load go/1.19.1
module load singularity/3.10.2

# ("sic" "osisaf_trend_5/sic_trend" "lsmask" "t2m" "xwind" "ywind")
declare -a permute1=("sic" "osisaf_trend_5/sic_trend" "lsmask" "xwind" "ywind")
declare -a permute2=("sic" "osisaf_trend_5/sic_trend" "lsmask" "t2m" "ywind")
declare -a permute3=("sic" "osisaf_trend_5/sic_trend" "lsmask" "t2m" "xwind")
declare -a permute4=("sic" "osisaf_trend_5/sic_trend" "lsmask")
declare -a permute5=("sic" "osisaf_trend_5/sic_trend" "lsmask" "t2m")
declare -a permute6=("sic" "osisaf_trend_5/sic_trend" "t2m" "xwind" "ywind")
declare -a permute7=("sic" "lsmask" "t2m" "xwind" "ywind")
declare -a permute8=("osisaf_trend_5/sic_trend" "lsmask" "t2m" "xwind" "ywind")

declare -a permutes=("${permute1[*]@Q}" "${permute2[*]@Q}" "${permute3[*]@Q}" "${permute4[*]@Q}" "${permute5[*]@Q}" "${permute6[*]@Q}" "${permute7[*]@Q}" "${permute8[*]@Q}")

for permute in "${permutes[@]}"; do
	echo "$permute"
	COMMAND="python -u /mnt/SimpleUNET/RunModel/permute_run_model.py ${permute}"
	singularity exec -B /lustre/storeB/users/arefk/MScThesis_AreKvanum2022_SeaIceML:/mnt --nv $HOME/TFcontainer/tensorflow_latest-gpu.sif bash -c "$COMMAND > /mnt/SimpleUNET/RunModel/logs/$(date -d "today" +"%d%m%H%M").log 2> /mnt/SimpleUNET/RunModel/errors/$(date -d "today" +"%d%m%H%M").log"
done
