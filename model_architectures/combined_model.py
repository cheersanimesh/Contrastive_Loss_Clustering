import base_cnn
import clustering_head
import contrastive_head
import keras
import tensorflow as tf
import sys
from tqdm import tqdm
import datetime
from tensorflow.keras import optimizers

sys.path.append(0,'-----vars directory-----')
import vars
import model_architectures.logger as logger
import loss_functions.clustering_loss as cl_loss
import loss_functions.contrastive_loss as co_loss

class combined_model:
    
    def __init__(self):
        base_cnn_obj= base_cnn(vars.input_shape)
        self.time_stamp= str(datetime.datetime.now())
        clustering_head_obj = clustering_head(input_shape= vars.clustering_head_input, output_shape=vars.clustering_head_output, batch_size= vars.batch_size )
        contrastive_head_obj = contrastive_head(input_shape= vars.contrastive_head_input, output_shape= vars.contrastive_head_output, batch_size= vars.batch_size)

        self.base_cnn= base_cnn_obj.get_base_cnn(resnet_arch= vars.resnet_arch, output_dim= vars.base_cnn_output_dimension, weight_initialization= vars.base_cnn_weight_initialization)
        self.contrastive_head= contrastive_head_obj.get_contrastive_head(contrastive_head_mode= vars.contrastive_head_mode)
        self.clustering_head= clustering_head_obj.get_clustering_head(clustering_head_mode= vars.clustering_head_mode)
        self.clustering_loss= cl_loss.Clustering_loss()
        self.contrastive_loss = co_loss.Contrastive_loss()

    def call(self, data):
        return self.base_cnn(data)
    
    def call_contrastive(self, data):
        base_cnn_output= self.call(data)
        return self.contrastive_head(base_cnn_output)
    
    def call_clustering(self, data):
        base_cnn_output= self.call(data)
        return self.clustering_head(base_cnn_output)
    
    def compile(self, optimizer=vars.training_optimizer):
        self.optimizer= optimizer
    
    def fit(self, dataset):

        logger.logger_single_write(f"logs/loss_logs {self.time_stamp}.txt",'w'," starting ")
        logger.logger_single_write(f"logs/contrastive_logs {self.time_stamp}.txt","w","writing losses : \n")
        logger.logger_single_write(f"logs/clustering_logs {self.time_stamp}.txt","w","writing losses: \n")

        for epochs in range(vars.training_epochs):
            logger.logger_single_write(f"logs/loss_logs {self.time_stamp}.txt","a",f"Starting Epoch{epochs}: \n \n")
            logger.logger_single_write(f"logs/contrastive_logs {self.time_stamp}.txt","a",f"Starting Epoch{epochs}: \n \n")
            logger.logger_single_write(f"logs/clustering_logs {self.time_stamp}.txt","a",f"Starting Epoch{epochs}: \n \n")

            for idx, mini_batch in tqdm(enumerate(dataset)):
                loss= self.__train_step(mini_batch)
                logger.logger_multi_write(f"logs/loss_logs {self.time_stamp}.txt",'a',["  ||  "+str(loss['loss'].numpy())+"   ||  ","\n"])

    def __train_step(self, data):

        with tf.GradientTape() as tape:
            type_1_embeddings_contrastive= self.call_contrastive(data[0])
            type_2_embeddings_contrastive= self.call_contrastive(data[1])

            type_1_embeddings_clustering= self.call_clustering(data[0])
            type_2_embeddings_clustering= self.call_clustering(data[1])
            
            contrastive_loss= self.contrastive_loss.compute_loss(type_1_embeddings_contrastive, type_2_embeddings_contrastive)
            clustering_loss= self.clustering_loss.compute_loss(type_1_embeddings_clustering, type_2_embeddings_clustering)

            loss = contrastive_loss + clustering_loss

        logger.logger_single_write(f"logs/contrastive_logs {self.time_stamp}.txt",'a',f"|| {contrastive_loss} ||")
        logger.logger_single_write(f"logs/clustering_logs {self.time_stamp}.txt",'a',f"|| {clustering_loss} ||")

        print(f"Clustering Loss -->   {clustering_loss}")
        print(f"Contrastive Loss -->  {contrastive_loss}")

        gradients= tape.gradient(loss, {'base_cnn':self.base_network.trainable_weights,
                                      'contrastive_level':self.contrastive_head.trainable_weights,
                                      'clustering_level':self.clustering_head.trainable_weights})

        self.optimizer.apply_gradients(
            zip(gradients['base_cnn'], self.base_cnn.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['contrastive_level'], self.contrastive_head.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['clustering_level'], self.clustering_head.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def save_models(self):
        self.base_cnn.save(f"../model_checkpoints/base_cnn {self.time_stamp}.h5")
        self.contrastive_head.save(f"../model_checkpoints/contrastive_head {self.time_stamp}.h5")
        self.clustering_head.save(f"../model_checkpoints/clustering_head {self.time_stamp}.h5")
    

    def kmeans_centres(self, embeddings, k):
        ## inputs embeddings returns KMEANS centres/centroids with number of centroids =k 

        ## ------Insert the code below ---------
        k_means_centres= np.array([]) 
        return k_means_centres

    def finch_centres(self, embeddings):
        ## inputs embeddings and returns the centres/partitions returned if FINCH algorithm
        ## stopping criteria may be assumed as required
        ## ------Insert the code below ---------

        finch_centres = np.array([])
        return finch_centres
    


        
    
    



