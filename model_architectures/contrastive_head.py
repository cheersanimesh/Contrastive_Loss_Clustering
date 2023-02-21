from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

class contrastive_mlp:

    def __init__(self, input_shape, output_shape, batch_size):
        self.input_shape= input_shape
        self.output_shape= output_shape
        self.batch_size= batch_size

    def get_contrastive_head(self, contrastive_head_mode):

        contrastive_mlp=tf.keras.Sequential()
        contrastive_mlp.add(layers.Dense(self.input_shape))
        contrastive_mlp.add(layers.BatchNormalization())
        contrastive_mlp.add(layers.Dense(64))
        contrastive_mlp.add(layers.BatchNormalization())
        contrastive_mlp.add(layers.Dense(32))
        contrastive_mlp.add(layers.Dense(16))
        contrastive_mlp.add(layers.Dense(self.output_shape)) 
    
        contrastive_mlp.build(input_shape=(self.batch_size,self.input_shape))

        return contrastive_mlp