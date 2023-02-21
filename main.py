import tensorflow as tf
import keras
import numpy as np
import vars
import sys
import dataset_loader.load_dataset as ld_data
import model_architectures.combined_model as cm_model

import model_architectures.logger as logger
import testing_models.testing as testing

print("Starting Execution")

print("Loading Dataset")

dataset= ld_data.dataset(vars.dataset)

training_dataset, val_dataset = dataset.get_train_dataset(train_val_split= vars.train_test_split, batch_size=vars.batch_size)

testing_dataset, test_labels= dataset.get_test_dataset(batch_size=vars.batch_size)

print("Initializing model")

contrastive_clustering_model = cm_model.combined_model()

timestamp= contrastive_clustering_model.time_stamp

print("Compiling model")

contrastive_clustering_model.compile(optimizer= vars.optimizers)

print("Training Model")
contrastive_clustering_model.fit(training_dataset)

print("Save Model")
contrastive_clustering_model.save_models()

print("Testing Model")
base_cnn= contrastive_clustering_model.base_cnn
clustering_head= contrastive_clustering_model.clustering_head

print("Testing Model")

tester= testing.testing(base_cnn=base_cnn, clustering_head=clustering_head, timestamp=timestamp)

predictions = tester.get_predictions(test_dataset=testing_dataset)

scores = tester.get_scores(predictions=predictions, y_test=predictions)

nmi_score= scores[0]
rand_score= scores[1]
fmi_score= scores[2]

print("NMI SCORE --> ")
print(normalized_mutual_info_score(y_test,predictions))

print("Adjusted Rand Score -->  ")
print(adjusted_rand_score(y_test, predictions))

print("Fowlkes Mallows Score --> ")
print(fowlkes_mallows_score (y_test, predictions))

logger.logger_multi_write(f'Results/Predictions {timestamp} .txt','a',['Predictions --> ', str(predictions)])
logger.logger_multi_write(f'Results/Actual Labels {timestamp} .txt','a',['Actual Labels --> ', str(predictions)])

logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'NMI Score --> {nmi_score} \n')
logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'Rand Score --> {rand_score} \n')
logger.logger_single_write(f'Results/Scores {timestamp} .txt','a',f'Fmi Score --> {fmi_score} \n')














