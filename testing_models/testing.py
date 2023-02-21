import tensorflow as tf
import numpy as np
import keras
import vars
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score

class testing:

    def __init__(self, base_cnn, clustering_head, timestamp):
        self.base_cnn= base_cnn
        self.clustering_head= clustering_head
        self.timestamp= timestamp

    def get_predictions(self,test_dataset):

        predictions=[]
        for mini_batch in test_dataset:
            embeddings_base_cnn= self.base_cnn(mini_batch) 
            embeddings= self.clustering_head(embeddings_base_cnn)
            predictions.append(tf.math.argmax(embeddings).numpy())
        
        return predictions
    
    def get_scores(self,predictions, y_test):
        nmi_score= normalized_mutual_info_score(y_test,predictions)
        rand_score = adjusted_rand_score(y_test, predictions)
        fmi_score = fowlkes_mallows_score (y_test, predictions)

        return [nmi_score, rand_score, fmi_score]
        
    


