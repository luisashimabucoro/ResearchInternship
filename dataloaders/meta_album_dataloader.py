import os
import argparse
import pandas as pd
import numpy as np
import pickle
import yaml


class MetaAlbumDataloader(object):

    def __init__(self, dataset_dir, target_dir, seed, N_way):
        self.dataset_dir = dataset_dir 
        self.target_dir = target_dir
        self.N_way = N_way
        self.seed = seed
    
    def set_seed(self):
        np.random.seed(self.seed)

    def train_test_split_classes(self, classes, test_split=0.3):
        self.set_seed()

        classes = np.unique(np.array(classes))
        np.random.shuffle(classes)
        threshold = int((1-test_split)*len(classes))

        return classes[:threshold], classes[threshold:]
    
    def create_split_csv(self, dataset_id, folder_name):
        """
        Function used to create train, val and test .csvs out of a single csv
        
        """
        dataset_csv = pd.read_csv(f"{self.target_dir}/{dataset_id}/{folder_name}/labels.csv", header=0, sep=',')
        total_classes = dataset_csv['CATEGORY']
        
        # default split: 70/15/15
        train_classes, test_classes = self.train_test_split_classes(total_classes, test_split=0.3) 
        test_classes, val_classes = self.train_test_split_classes(test_classes, test_split=0.5) 
        
        # split for too little classes: 60/20/20
        if len(val_classes) + len(test_classes) < 2*self.N_way:
            train_classes, val_classes = self.train_test_split_classes(total_classes, test_split=0.4)
            val_classes, test_classes = self.train_test_split_classes(val_classes, test_split=0.5)

        # backup split: varied
        if len(val_classes) <= self.N_way or len(test_classes) <= self.N_way:
            min_percentage = 2*(self.N_way) / len(np.unique(self.N_way))
            train_classes, val_classes = self.train_test_split_classes(total_classes, test_split=min_percentage)
            test_classes, val_classes = self.train_test_split_classes(val_classes, test_split=0.5)

        short_dataset = dataset_csv[['FILE_NAME', 'CATEGORY']].rename(columns={'FILE_NAME' : 'filename', 'CATEGORY' : 'label'})

        train_dataset = short_dataset[short_dataset['label'].isin(train_classes)].copy()
        train_dataset.to_csv(path_or_buf=f"{self.target_dir}/{dataset_id}/{folder_name}/train.csv", index=False)

        test_dataset = short_dataset[short_dataset['label'].isin(test_classes)].copy()
        test_dataset.to_csv(path_or_buf=f"{self.target_dir}/{dataset_id}/{folder_name}/test.csv", index=False)

        val_dataset = short_dataset[short_dataset['label'].isin(val_classes)].copy()
        val_dataset.to_csv(path_or_buf=f"{self.target_dir}/{dataset_id}/{folder_name}/val.csv", index=False)

        print(f'Split files created for dataset {folder_name}!')


def train_test_split_classes(classes, test_split=0.3, seed=42):
    np.random.seed(seed)

    classes = np.unique(np.array(classes))
    np.random.shuffle(classes)
    threshold = int((1-test_split)*len(classes))

    return classes[:threshold], classes[threshold:]

def merge_datasets(dataset_paths, target_dir):
    df = pd.DataFrame(columns=['filename', 'label'])
    for dataset_path in dataset_paths:
        dataset = pd.read_csv(f"{dataset_path}/labels.csv").rename(columns={'FILE_NAME' : 'filename', 'CATEGORY' : 'label'})
        rel_path = os.path.relpath(dataset_path, f"{target_dir}/images")
        dataset['filename'] = rel_path + '/images/' + dataset['filename'].str[:]
        df = pd.concat([df, dataset[['filename', 'label']]], axis=0)
    
    return df

def save_split_csv(dataset_paths, target_dir, split='train'):
    df = merge_datasets(dataset_paths, target_dir)
    classes, _ = train_test_split_classes(df['label'], test_split=0)
    dataset = df[df['label'].isin(classes)].copy()
    dataset.to_csv(path_or_buf=f"{target_dir}/{split}.csv", index=False)

    print(f"{split} dataset successfully created!")

def create_cross_domain_split(train_datasets, val_datasets, test_datasets, target_dir):
    """
    Function to create custom crss-domain split dataset

    train_datasets : list of train dataset paths
    test_datasets : list of test dataset paths
    target_dir : directory where train/val/test csvs should be stored
    """
    # creating target directory if it doesnt already exist
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        os.mkdir(f"{target_dir}/images")
        print(f"Target directory {target_dir} created!")
    
    save_split_csv(train_datasets, target_dir, split='train')
    save_split_csv(val_datasets, target_dir, split='val')
    save_split_csv(test_datasets, target_dir, split='test')
    
    print("Cross domain split files successfully created!")

def main():
    parser = argparse.ArgumentParser(description='Argument parser for Meta Album dataloader')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-dir', type=str, default='/home/lushimabucoro/.cache/openml/org/openml/www/datasets')
    parser.add_argument('--target-dir', type=str, default='/home/lushimabucoro/.cache/openml/org/openml/www/datasets')
    args = parser.parse_args()

    # dataset_dict = {44320: 'BRD_Extended', 44317: 'PLK_Extended', 44318: 'FLW_Extended', 44321: 'PLT_VIL_Extended', 44316: 'BCT_Extended', 44324: 'RESISC_Extended', 44323: 'CRS_Extended', 44322: 'TEX_Extended', 44319: 'SPT_Extended', 44287: 'MD_MIX_Mini', 44331: 'DOG_Extended', 44326: 'INS_2_Extended', 44327: 'PLT_NET_Extended', 44332: 'MED_LF_Extended', 44330: 'PNU_Extended', 44333: 'RSICB_Extended', 44329: 'APL_Extended', 44328: 'TEX_DTD_Extended', 44325: 'ACT_40_Extended', 44296: 'MD_5_BIS_Mini', 44338: 'AWA_Extended', 44340: 'INS_Extended', 44335: 'FNG_Extended', 44336: 'PLT_DOC_Extended', 44342: 'PRT_Extended', 44341: 'RSD_Extended', 44343: 'BTS_Extended', 44337: 'TEX_ALOT_Extended', 44334: 'ACT_410_Extended', 44310: 'MD_6_Mini'}
    # if not os.path.isfile('/home/lushimabucoro/Codes/ResearchInternship/BEPE/dataloaders/utils/meta_album_dict.pickle'):
    #     with open('/home/lushimabucoro/Codes/ResearchInternship/BEPE/dataloaders/utils/meta_album_dict.pickle', 'wb') as metadata:
    #         pickle.dump(dataset_dict, metadata)

    # meta_album_dict = pickle.load(open('/home/lushimabucoro/Codes/ResearchInternship/BEPE/dataloaders/utils/meta_album_dict.pickle', 'rb'))
    

    # dataloader = MetaAlbumDataloader(args.dataset_dir, args.target_dir, args.seed, 5)
    # for dataset_id, folder_name in meta_album_dict.items():
    #     dataloader.create_split_csv(str(dataset_id), folder_name)
    with open("cross_domain.yaml", 'r') as stream:
        config_dict = yaml.safe_load(stream)
    
    create_cross_domain_split(config_dict['train_datasets'], config_dict['test_datasets'], config_dict['target_dir'])
        

if __name__ == '__main__':
    main()