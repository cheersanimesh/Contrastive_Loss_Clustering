import tensorflow as tf
import keras
import numpy as np
import vars
import sys
import os
from tensorflow import keras
import datetime

parrent_path= os.getcwd()
'''
with open('a.txt', 'r') as f:
	directories= f.readlines()

for dir in directories:
	sys.path.append(parrent_path+dir.strip())
print(sys.path)
'''
directories_list = ['Results','__pycache__','dataset_dump','dataset_loader','logs','loss_functions','model_architectures','model_checkpoints','testing_models']

for name_dir in directories_list:
    dir_path= os.path.join(parrent_path, name_dir)
    sys.path.append(dir_path)

##sys.path.append("/home/csb1051719/ContrastiveClustering/Contrastive_Loss_Clustering/dataset_loader")
timestamp = str(datetime.datetime.now())
import dataset_loader.load_dataset as ld_data
import model_architectures.combined_model as cm_model

import model_architectures.logger as logger
import testing_models.testing as testing
from sklearn.metrics.cluster import normalized_mutual_info_score, fowlkes_mallows_score, adjusted_rand_score

print("Starting Execution")

print("Loading Dataset")

dataset= ld_data.dataset(vars.dataset)

training_dataset, val_dataset = dataset.get_train_dataset(train_val_split= vars.train_test_split, batch_size=vars.batch_size)

testing_dataset, test_labels= dataset.get_test_dataset(batch_size=vars.batch_size)

print("loading model")

print("enter base_cnn path")
base_cnn_path = str(input())
print("clustering head path")
clustering_head_path = str(input())
base_cnn_path= os.path.join(parrent_path,'model_checkpoints',base_cnn_path)
clustering_head_path= os.path.join(parrent_path, 'model_checkpoints',clustering_head_path)

base_cnn= keras.models.load_model(base_cnn_path)
clustering_head= keras.model.load_model(clustering_head_path)

print("Testing Model")

tester= testing.testing(base_cnn=base_cnn, clustering_head=clustering_head, timestamp= timestamp)

predictions = tester.get_predictions(test_dataset=testing_dataset)

scores = tester.get_scores(predictions=predictions, y_test=predictions)

nmi_score= scores[0]
rand_score= scores[1]
fmi_score= scores[2]
print(nmi_score)
print(rand_score)
print(fmi_score)
test_labels = np.squeeze(test_labels)
print(predictions.shape)

print("NMI SCORE --> ")
print(normalized_mutual_info_score(test_labels,predictions))

print("Adjusted Rand Score -->  ")
print(adjusted_rand_score(test_labels, predictions))

print("Fowlkes Mallows Score --> ")
print(fowlkes_mallows_score (test_labels, predictions))

logger.logger_multi_write(f'Results/Predictions {timestamp} .txt','a',['Predictions --> ', str(predictions)])
logger.logger_multi_write(f'Results/Actual Labels {timestamp} .txt','a',['Actual Labels --> ', str(predictions)])

logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'NMI Score --> {nmi_score} \n')
logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'Rand Score --> {rand_score} \n')
logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'Fmi Score --> {fmi_score} \n')