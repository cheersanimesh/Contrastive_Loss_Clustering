from tensorflow.keras import optimizers

batch_size= 32
target_shape=(200200)
num_classes =10
dataset='cifar_100'
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

contrastive_head_input=128
contrastive_head_output=8
clustering_head_input=128
clustering_head_output=8
training_epochs=100

centre_initilization_mode='finch'
initial_training_epochs=100
next_training_epochs= 100

training_optimizer= optimizers.Adam(0.0001,epsilon=0.1)






