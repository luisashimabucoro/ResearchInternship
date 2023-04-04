#!/bin/bash

# Baseline
echo $'########################################################'
echo $'###--- Within-Domain Few-Shot Learning Experiments ---###'
echo $'########################################################\n\n'

Shots=(3 2 1)

for shot in ${Shots[@]}; do 
    echo "Training" $shot "shot model"
    python3 run_trainer.py --shot_num $shot --data_root $1 --device_ids $2
done