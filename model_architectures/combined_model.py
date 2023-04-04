from base_cnn import base_cnn
from clustering_head import clustering_mlp
from contrastive_head import contrastive_mlp
import keras
import tensorflow as tf
import sys
from tqdm import tqdm
import datetime
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from sklearn.cluster import KMeans
from finch import FINCH
import numpy as np

# sys.path.append(0,'-----vars directory-----')
import vars
import model_architectures.logger as logger
import loss_functions.clustering_loss as cl_loss
import loss_functions.contrastive_loss as co_loss


class combined_model:

    def __init__(self):
        base_cnn_obj = base_cnn(vars.input_shape)
        self.time_stamp = str(datetime.datetime.now())
        clustering_head_obj = clustering_mlp(
            input_shape=vars.clustering_head_input, output_shape=vars.clustering_head_output, batch_size=vars.batch_size)
        contrastive_head_obj = contrastive_mlp(
            input_shape=vars.contrastive_head_input, output_shape=vars.contrastive_head_output, batch_size=vars.batch_size)

        self.base_cnn = base_cnn_obj.get_base_cnn(
            resnet_arch=vars.resnet_arch, output_dim=vars.base_cnn_output_dimension, weight_initialization=vars.base_cnn_weight_initialization)
        self.contrastive_head = contrastive_head_obj.get_contrastive_head(
            contrastive_head_mode=vars.contrastive_head_mode)
        self.clustering_head = clustering_head_obj.get_clustering_head(
            clustering_head_mode=vars.clustering_head_mode)
        self.clustering_loss = cl_loss.Clustering_loss()
        self.contrastive_loss = co_loss.Contrastive_loss()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, data):
        return self.base_cnn(data)

    def call_contrastive(self, data):
        base_cnn_output = self.call(data)
        return self.contrastive_head(base_cnn_output)

    def call_clustering(self, data):
        base_cnn_output = self.call(data)
        return self.clustering_head(base_cnn_output)

    def compile(self, optimizer=vars.training_optimizer):
        self.optimizer = optimizer

    def fit(self, dataset):

        logger.logger_single_write(
            f"logs/loss_logs {self.time_stamp}.txt", 'w', " starting ")
        logger.logger_single_write(
            f"logs/contrastive_logs {self.time_stamp}.txt", "w", "writing losses : \n")
        logger.logger_single_write(
            f"logs/clustering_logs {self.time_stamp}.txt", "w", "writing losses: \n")

        logger.logger_single_write(
            f"logs/loss_logs {self.time_stamp}.txt", 'w', " initial_stage ")
        logger.logger_single_write(
            f"logs/contrastive_logs {self.time_stamp}.txt", "w", "inital_stage : \n")

        for epochs in range(vars.initial_training_epochs):
            logger.logger_single_write(
                f"logs/loss_logs {self.time_stamp}.txt", "a", f"Starting Epoch{epochs}: \n \n")
            logger.logger_single_write(
                f"logs/contrastive_logs {self.time_stamp}.txt", "a", f"Starting Epoch{epochs}: \n \n")

            for idx, mini_batch in tqdm(enumerate(dataset)):
                loss = self.__train_step_initial(mini_batch)
                logger.logger_multi_write(f"logs/loss_logs {self.time_stamp}.txt", 'a', [
                                          "  ||  "+str(loss['loss'].numpy())+"   ||  ", "\n"])

        embeddings = self.__get_embeddings(dataset=dataset)
        self.cluster_centres = self.return_cluster_centres(embeddings)

        logger.logger_single_write(
            f"logs/loss_logs {self.time_stamp}.txt", 'w', " \n second_stage ")
        logger.logger_single_write(
            f"logs/contrastive_logs {self.time_stamp}.txt", "w", "\n second_stage : \n")
        logger.logger_single_write(
            f"logs/clustering_logs {self.time_stamp}.txt", "w", "\n second_stage : \n")

        for epochs in range(vars.training_epochs):
            logger.logger_single_write(
                f"logs/loss_logs {self.time_stamp}.txt", "a", f"Starting Epoch{epochs}: \n \n")
            logger.logger_single_write(
                f"logs/contrastive_logs {self.time_stamp}.txt", "a", f"Starting Epoch{epochs}: \n \n")
            logger.logger_single_write(
                f"logs/clustering_logs {self.time_stamp}.txt", "a", f"Starting Epoch{epochs}: \n \n")

            for idx, mini_batch in tqdm(enumerate(dataset)):
                loss = self.__train_step(mini_batch)
                logger.logger_multi_write(f"logs/loss_logs {self.time_stamp}.txt", 'a', [
                                          "  ||  "+str(loss['loss'].numpy())+"   ||  ", "\n"])

    def return_cluster_centres(self, data):
        data_numpy = data.numpy()
        centres = []
        if (vars.centre_initilization_mode == 'finch'):
            centres = self.finch_centres(data_numpy)
        if (vars.centre_initilization_mode == 'kmeans'):
            centes = self.kmeans_centres(data_numpy)

        return tf.Variable(centres, dtype=tf.float32)

    def __train_step_initial(self, data):

        with tf.GradientTape() as tape:
            type_1_embeddings_contrastive = self.call_contrastive(data[0])
            type_2_embeddings_contrastive = self.call_contrastive(data[1])
            contrastive_loss = self.contrastive_loss.compute_loss(
                type_1_embeddings_contrastive, type_2_embeddings_contrastive)

            loss = contrastive_loss

        logger.logger_single_write(
            f"logs/contrastive_logs {self.time_stamp}.txt", 'a', f"|| {contrastive_loss} ||")

        print(f"Contrastive Loss -->  {contrastive_loss}")

        gradients = tape.gradient(loss, {'base_cnn': self.base_cnn.trainable_weights,
                                         'contrastive_level': self.contrastive_head.trainable_weights})

        self.optimizer.apply_gradients(
            zip(gradients['base_cnn'], self.base_cnn.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['contrastive_level'],
                self.contrastive_head.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def __train_step(self, data, cluster_centres=None):

        with tf.GradientTape() as tape:
            type_1_embeddings_contrastive = self.call_contrastive(data[0])
            type_2_embeddings_contrastive = self.call_contrastive(data[1])

            type_1_embeddings_clustering = self.call_clustering(data[0])
            type_2_embeddings_clustering = self.call_clustering(data[1])

            contrastive_loss = self.contrastive_loss.compute_loss(
                type_1_embeddings_contrastive, type_2_embeddings_contrastive)
            clustering_loss = self.clustering_loss.compute_loss(
                type_1_embeddings_clustering, type_2_embeddings_clustering, cluster_centres=self.cluster_centres)

            loss = contrastive_loss + clustering_loss

        logger.logger_single_write(
            f"logs/contrastive_logs {self.time_stamp}.txt", 'a', f"|| {contrastive_loss} ||")
        logger.logger_single_write(
            f"logs/clustering_logs {self.time_stamp}.txt", 'a', f"|| {clustering_loss} ||")

        print(f"Clustering Loss -->   {clustering_loss}")
        print(f"Contrastive Loss -->  {contrastive_loss}")

        gradients = tape.gradient(loss, {'base_cnn': self.base_cnn.trainable_weights,
                                         'contrastive_level': self.contrastive_head.trainable_weights,
                                         'clustering_level': self.clustering_head.trainable_weights,
                                         'cluster_centres': self.cluster_centres})

        self.optimizer.apply_gradients(
            zip(gradients['base_cnn'], self.base_cnn.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['contrastive_level'],
                self.contrastive_head.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['clustering_level'],
                self.clustering_head.trainable_weights)
        )
        self.optimizer.apply_gradients(
            zip(gradients['cluster_centres'], self.cluster_centres)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def save_models(self):
        self.base_cnn.save(
            f"../model_checkpoints/base_cnn {self.time_stamp}.h5")
        self.contrastive_head.save(
            f"../model_checkpoints/contrastive_head {self.time_stamp}.h5")
        self.clustering_head.save(
            f"../model_checkpoints/clustering_head {self.time_stamp}.h5")

    def kmeans_centres(self, embeddings, k):
        # inputs embeddings returns KMEANS centres/centroids with number of centroids =k

        # ------Insert the code below ---------
        kmeans = KMeans(n_clusters=k).fit(embeddings)
        centroids = kmeans.cluster_centers_
        return centroids

    def finch_centres(self, embeddings):
        # inputs embeddings and returns the centres/partitions returned if FINCH algorithm
        # stopping criteria may be assumed as required
        # ------Insert the code below ---------
        c, num_clust, req_c = FINCH(embeddings)
        return (c, num_clust, req_c)

    def __get_embeddings(self, dataset):

        embeddings = []
        for mini_batch in dataset:
            embeddings_base_cnn_1 = self.base_cnn(mini_batch[0])
            embeddings_clustering_1 = self.clustering_head(
                embeddings_base_cnn_1)
            embeddings_base_cnn_2 = self.base_cnn(mini_batch[1])
            embeddings_clustering_2 = self.clustering_head(
                embeddings_base_cnn_2)
            for embed in embeddings_clustering_1:
                embeddings.append(embed.numpy())

            for embed in embeddings_clustering_2:
                embeddings.append(embed.numpy())
        embeddings = np.array(embeddings)

        return embeddings
