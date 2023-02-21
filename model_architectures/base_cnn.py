from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow.keras import Model

class base_cnn():

    def __init__(self, input_shape):
        self.input_shape= input_shape
        
        print("base_cnn initializing")

    def get_base_cnn(self,resnet_arch, output_dim, weight_initialization,base_cnn_mode='def'):

        base_cnn = resnet.ResNet50(
                weights=weight_initialization, input_shape=self.input_shape + (3,), include_top=False
            )
        if(resnet_arch=='resnet_50'):
            base_cnn = resnet.ResNet50(
                weights=weight_initialization, input_shape=self.input_shape + (3,), include_top=False
            )

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2= layers.BatchNormalization()(dense2)
        output = layers.Dense(output_dim)(dense2)

        embedding_layer = Model(base_cnn.input, output, name="Embedding")

        return embedding_layer
