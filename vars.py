from tensorflow.keras import optimizers

batch_size= 64
target_shape=(200,200)
num_classes =10
dataset='cifar_10'
augmentation_mode=''
train_test_split= 0.8

num_clusters=10
base_cnn_output_dimension = 128
base_cnn_weight_initialization=None
contrastive_head_mode=''
clustering_head_mode=''
contrastive_loss_function_mode=''
clustering_loss_function_mode=''
contrastive_loss_temperature=0.5
clustering_loss_temperature=1.0
input_shape=(200, 200)

contrastive_head_input=128
contrastive_head_output=8
clustering_head_input=128
clustering_head_output=num_clusters
training_epochs=5

initial_training_epochs=5
centre_initilization_mode='kmeans'
next_training_epochs= 5

training_optimizer= optimizers.legacy.Adam(learning_rate=0.001,epsilon=0.1)
resnet_arch= "resnet_50"





