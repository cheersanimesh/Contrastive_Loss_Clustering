import vars
import os

parrent_path= os.getcwd()

dataset_name = vars.dataset

dataset_path= os.path.join(parrent_path,"dataset_dump")

dataset_path= os.path.join(dataset_path, dataset_name)

if(not os.path.exists(dataset_path)):
    print("creating directory")
    os.mkdir(dataset_path)