from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

class clustering_mlp:

    def __init__(self, input_shape, output_shape, batch_size):
        self.input_shape= input_shape
        self.output_shape= output_shape
        self.batch_size= batch_size

    def get_clustering_head(self, clustering_head_mode):

        clustering_mlp=tf.keras.Sequential()
        clustering_mlp.add(layers.Dense(self.input_shape))
        clustering_mlp.add(layers.BatchNormalization())
        clustering_mlp.add(layers.Dense(64))
        clustering_mlp.add(layers.BatchNormalization())
        clustering_mlp.add(layers.Dense(32))
        clustering_mlp.add(layers.Dense(16))
        clustering_mlp.add(layers.Dense(self.output_shape)) 
    
        clustering_mlp.build(input_shape=(self.batch_size,self.input_shape))

        return clustering_mlp