import vars
import os

parrent_path= os.getcwd()

dataset_name = vars.dataset

dataset_path= os.path.join(parrent_path,"dataset_dump")
train_dataset_path=os.path.join(dataset_path,"train",dataset_name)

test_dataset_path= os.path.join(dataset_path, "test", dataset_name)

if(not os.path.exists(train_dataset_path)):
    print("creating training directory")
    os.mkdir(train_dataset_path)

if(not os.path.exists(test_dataset_path)):
    print("creating test directory")
    os.mkdir(test_dataset_path)