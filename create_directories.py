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
