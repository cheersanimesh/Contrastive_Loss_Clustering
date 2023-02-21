import tensorflow as tf

class Clustering_loss:
  
    def __init__(self, mode='Standard_KMeans'):
      self.mode=mode
    
    def compute_loss(self, hidden1, hidden2, temperature= 0.5):

        if(mode=='Standard_KMeans'):

            ##Dummy Code

            weights=1.0
            LARGE_NUM=1e9
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
            masks = tf.one_hot(tf.range(batch_size), batch_size) 

            logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
            logits_aa = logits_aa - masks * LARGE_NUM
            logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
            logits_bb = logits_bb - masks * LARGE_NUM
            logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
            logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

            loss_a = tf.nn.softmax_cross_entropy_with_logits(
                labels, tf.concat([logits_ab, logits_aa], 1))
                
            loss_b = tf.nn.softmax_cross_entropy_with_logits(
                labels, tf.concat([logits_ba, logits_bb], 1))
            
            loss = tf.reduce_mean(loss_a + loss_b)

            return loss