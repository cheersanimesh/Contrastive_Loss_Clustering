import keras
import cv2
import tensorflow as tf
from glob import glob
import dataset_operations
import numpy as np

class dataset:
    def __init__(self, dataset):

        if(dataset=='cifar_10'):
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            np.random.shuffle(x_train)

            ctr=0
            for img in x_train:
                cv2.imwrite(f'../dataset_dump/train/{dataset}/{ctr}.jpg', img)
                ctr+=1
            ctr=0
            for img in x_test:
                cv2.imwrite(f'../dataset_dump/test/{dataset}/{ctr}.jpg', img)
                ctr+=1
        self.training_paths= glob(f'../dataset_dump/train/{dataset}/*.jpg')
        self.testing_paths= glob(f'../dataset_dump/test/{dataset}/*.jpg')
        self.testing_labels= y_test
    
    def get_train_dataset(self, train_val_split, batch_size):
        image_paths= self.training_paths
        dataset_len= len(image_paths)
        image_count = dataset_len

        dataset_a = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset_b = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset_a = dataset_a.map(dataset_operations.read_preprocess_and_augment_image)
        dataset_b = dataset_b.map(dataset_operations.read_preprocess_and_augment_image)

        train_dataset = dataset.take(round(image_count * train_val_split))
        val_dataset = dataset.skip(round(image_count * train_val_split))

        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        self.train_dataset = train_dataset.prefetch(batch_size)

        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
        self.val_dataset = val_dataset.prefetch(batch_size)  

        return (self.train_dataset, self.val_dataset)

    def get_test_dataset(self, batch_size):
        image_paths= self.testing_paths

        test_dataset= tf.data.Dataset.from_tensor_slices(image_paths)
        test_dataset= test_dataset.map(dataset_operations.read_and_preprocess_image)
        test_dataset= test_dataset.batch(batch_size)

        return (test_dataset, self.testing_labels)
    
    
    
    
        
            

