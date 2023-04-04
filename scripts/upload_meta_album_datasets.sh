#!/bin/bash
local_dataset_dir="/home/lushimabucoro/.cache/openml/org/openml/www/datasets"
server_dataset_dir="invincible:/mnt/invinciblefs/scratch/lushima/datasets"

echo $'########################################################'
echo $'###---         Meta Album Datasets Upload         ---###'
echo $'########################################################\n\n'

Done_datasets=("44287/MD_MIX_Mini" "44296/MD_5_BIS_Mini" "44310/MD_6_Mini" "44316/BCT_Extended" "44318/FLW_Extended" "44319/SPT_Extended" "44322/TEX_Extended" "44323/CRS_Extended" "44324/RESISC_Extended" "44325/ACT_40_Extended" "44326/INS_2_Extended" "44327/PLT_NET_Extended" "44328/TEX_DTD_Extended" "44329/APL_Extended" "44330/PNU_Extended" "44331/DOG_Extended" "44332/MED_LF_Extended" "44333/RSICB_Extended" "44334/ACT_410_Extended" "44335/FNG_Extended" "44336/PLT_DOC_Extended" "44337/TEX_ALOT_Extended" "44338/AWA_Extended")
Datasets=("44320/BRD_Extended" "44321/PLT_VIL_Extended" "44341/RSD_Extended" "44342/PRT_Extended" "44343/BTS_Extended" "44340/INS_Extended" "44317/PLK_Extended")

for done_dataset in ${Done_datasets[@]}; do
    echo $'Uploading dataset .csvs '$done_dataset
    scp -r $local_dataset_dir/$done_dataset/*.csv $server_dataset_dir/$done_dataset/*.csv
done

for dataset in ${Datasets[@]}; do
    echo $'Uploading dataset '$dataset
    scp -r $local_dataset_dir/$dataset $server_dataset_dir
done