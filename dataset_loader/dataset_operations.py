import sys
sys.path.append('--vars_file_loc__')
import vars
import random
import tensorflow as tf

def read_preprocess_and_augment_image(filename):
    '''
      Accepts a file path and returns the image stored in the file path
    '''
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, vars.target_shape,method='bilinear')
    ##image = alter_image(image)
    aug_image = generate_augmentation(image)
    return aug_image

def read_and_preprocess_image(filename, target_shape):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape,method='bilinear')
    return image

def generate_augmentation(image):
    '''
     Inputs an image and returns its augmentation
    '''
    augmentations_mode=['Resized Crop','Random Brightness','Random_Contrast','Gaussian Blur']
    aug_mode_idx= random.randint(0,len(augmentations_mode))
    aug_image= image

    aug_image = tf.image.random_crop(image, size=[120,120,3])
    aug_image = tf.image.resize(aug_image, vars.target_shape, method='bilinear')

    aug_image= tf.image.random_brightness(aug_image,0.3)
    aug_image= tf.image.random_contrast(aug_image, 0.3, 0.7)
    aug_image= tf.image.random_jpeg_quality(aug_image,min_jpeg_quality=10, max_jpeg_quality = 100)
    return aug_image