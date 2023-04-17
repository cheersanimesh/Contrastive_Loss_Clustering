import tensorflow as tf
import vars

eps= 1e-8
class Clustering_loss:
  
    def __init__(self, mode='SCCL'):
      self.mode=mode
      self.alpha = 1.0
    
    def get_target_distb(self, batch):
        weight = (batch**2)/(tf.reduce_sum(batch, axis=0)+1e-9)
        output = tf.transpose(weight)/tf.transpose(tf.reduce_sum(weight,axis=1))
        return tf.transpose(output)

    def kl_div(self, predict, target):
        p1 = predict+eps
        t1= target+eps
        logP= tf.math.log(p1)
        logT= tf.math.log(t1)
        ##print(logP.shape)
        ##print(logT.shape)
        ##print(target.shape)
        #TlogTdI = tf.math.subtract(logT, logP)
        TlogTdI = tf.math.multiply(target, tf.math.subtract(logT, logP))
        kld= tf.reduce_sum(TlogTdI, axis=1)
        return tf.reduce_mean(kld)

    def get_cluster_prob(self,embeddings, cluster_centres):
        norm_squared= tf.reduce_sum((tf.expand_dims(embeddings,1)- cluster_centres)**2, axis=2)
        numr= 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numr= tf.math.pow(numr, power)
        output = numr / tf.reduce_sum(numr, axis=1, keepdims=True)
        return output

    def entropy_loss(self, probs1, probs2):
        
        clusterwise_probs_1 = tf.reduce_sum(probs1, axis=0)
        clusterwise_probs_2= tf.reduce_sum(probs2, axis=0)
        sum1= tf.reduce_sum(clusterwise_probs_1)
        sum2= tf.reduce_sum(clusterwise_probs_2)

        clusterwise_probs_1 = clusterwise_probs_1/sum1
        clusterwise_probs_2 = clusterwise_probs_2/sum2

        shape1= clusterwise_probs_1.shape[0]
        shape2= clusterwise_probs_2.shape[0]

        en_val1= tf.math.log(shape1)+tf.math.multiply(clusterwise_probs_1, tf.math.log(clusterwise_probs_1))
        en_val2= tf.math.log(shape2)+tf.math.multiply(clusterwise_probs_2, tf.math.log(clusterwise_probs_2))

        en1= tf.reduce_sum(en_val1)
        en2= tf.reduce_sum(en_val2)

        return en1+en2

    def compute_loss(self, hidden1, hidden2, cluster_centres=None):
        
        if(self.mode=='SCCL'):
            cluster_probs1= self.get_cluster_prob(hidden1, cluster_centres=cluster_centres)
            target1= self.get_target_distb(cluster_probs1)
            print(cluster_probs1.shape)
            print(target1.shape)
            cluster_loss_1 = self.kl_div(cluster_probs1, target1)/cluster_probs1.shape[0]

            cluster_probs2= self.get_cluster_prob(hidden2, cluster_centres=cluster_centres)
            target2= self.get_target_distb(cluster_probs2)
            cluster_loss_2 = self.kl_div(cluster_probs2, target2)/cluster_probs2.shape[0]

            entropy_loss= self.entropy_loss(cluster_probs1, cluster_probs2) 

            return ((cluster_loss_1+cluster_loss_2)/2.0) - entropy_loss
        




