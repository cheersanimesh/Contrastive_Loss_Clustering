import os
import sys


parrent_dir= os.getcwd()
sys.path.append(parrent_dir)
directories_list = ['Results','__pycache__','dataset_dump','dataset_loader','logs','loss_functions','model_architectures','model_checkpoints','testing_models']

for name_dir in directories_list:
    dir_path= os.path.join(parrent_dir, name_dir)
    if(not os.path.exists(dir_path)):
        os.mkdir(dir_path)
        sys.path.append(dir_path)

train_dataset_path= os.path.join(parrent_dir,'dataset_dump', 'train')
test_dataset_path= os.path.join(parrent_dir, 'dataset_dump','test')

if(not os.path.exists(train_dataset_path)):
    os.mkdir(train_dataset_path)

if(not os.path.exists(test_dataset_path)):
    os.mkdir(test_dataset_path)



