import os
import argparse
import pandas as pd
import numpy as np

"""
Script to modify miniImageNet dataset format in the following manner:
    - create train.csv, val.csv and test.csv
    - pass all the images into a single folder ./images
"""


def modify_dataset(dataset_dir, target_dir):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    
    for split in ['train', 'val', 'test']:
        split_file = open(os.path.join(dataset_dir, f'{split}.csv'), 'w')
        split_file.write('filename, label\n')
        for dir in os.listdir(os.path.join(dataset_dir, split)):
            print(dir)
            for file in os.listdir(os.path.join(dataset_dir, split, dir)):
                split_file.write(f'{file}, {dir}\n')
                os.rename(os.path.join(dataset_dir, split, dir, file), os.path.join(target_dir, file))
        
        split_file.close()


def main():
    parser = argparse.ArgumentParser(description="Arguments for dataset modification to suit LibFewShot")
    parser.add_argument('--dataset-dir', type=str, default='/home/lushimabucoro/datasets/miniImageNet') 
    parser.add_argument('--target-dir', type=str, default='/home/lushimabucoro/datasets/miniImageNet/images') 
    args = parser.parse_args()


    modify_dataset(args.dataset_dir, args.target_dir)

if __name__ == '__main__':
    main()