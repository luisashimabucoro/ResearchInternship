import openml
import os
import pickle

def download_datasets(dataset_dict, target_dir, rename_folders=True):
    """
    Funtion used to download meta album datasets and rename their folders (optional)

    dataset_dict : dictionary of format dataset_id : dataset_name
    """
    n_datasets = len(dataset_dict)
    for (idx, (dataset_id, dataset_name)) in enumerate(dataset_dict.items(), 1):
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)
        if rename_folders:
            os.rename(os.path.join(target_dir, str(dataset_id)), os.path.join(target_dir, dataset_name))

        print(f'{idx}/{n_datasets} datasets downloaded!')

def main():
    target_dir = '/home/lushimabucoro/.cache/openml/org/openml/www/datasets/'
    dataset_dict = {44320 : 'Birds', 44317 : 'Plankton', 44318 : 'Flowers', 44321 : 'Plant Village', 44316 : 'Bacteria', 44324 : 'RESISC', 44323 : 'Cars', 44322 : 'Textures', 44319 : '73 Sports', 44287 : 'Omniprint-MD-mix', 
                    44331 : 'Dogs', 44326 : 'Insects 2', 44327 : 'PlantNet', 44332 : 'Medicinal Leaf', 44330 : 'PanNuke', 44333 : 'RSICB', 44329 : 'Airplanes', 44328 : 'Textures DTD', 44325 : 'Stanford 40 Actions', 44296 : 'Omniprint-MD-5-bis', 
                    44338 : 'Animals with Attributes', 44340 : 'Insects', 44335 : 'Fungi', 44336 : 'PlantDoc', 44342 : 'Subcel. Human Protein', 44341 : 'RSD', 44343 : 'Boats', 44337 : 'Textures ALOT', 44334 : 'MPII Human Pose', 44310 : 'Omniprint-MD-6'}
    
    # creating dataset_pickle to store metadata
    with open('meta_album_dict.pickle', 'wb') as metadata:
        pickle.dump(dataset_dict, metadata)

    # download_datasets(dataset_dict, target_dir, rename_folders=False)

if __name__ == '__main__':
    main()