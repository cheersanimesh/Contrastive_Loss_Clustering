import tensorflow as tf
import vars

eps= 1e-8
class Clustering_loss:
  
    def __init__(self, mode='SCCL'):
      self.mode=mode
    
    def get_target_distb(self, batch):
        weight = (batch**2)/(tf.reduce_sum(batch, axis=0)+1e-9)
        return tf.transpose(weight)/tf.transpose(tf.reduce_sum(weight,axis=1))
    
    def kl_div(self, predict, target):
        p1 = predict+eps
        t1= target+eps
        logP= tf.math.log(p1)
        logT= tf.math.log(t1)
        TlogTdI = target * (logT - logP)
        kld= tf.reduce_sum(TlogTdI, axis=1)
        return tf.reduce_mean(kld)

    def get_cluster_prob(self,embeddings, cluster_centres):
        norm_squared= tf.reduce_sum((tf.expand_dims(embeddings,1)- cluster_centres)**2, axis=2)
        numr= 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        return numr / tf.reduce_sum(numr, axis=1, keepdims=True)

    def compute_loss(self, hidden1, hidden2, cluster_centres=None):
        
        if(self.mode=='SCCL'):
            cluster_probs1= self.get_cluster_prob(self, hidden1, cluster_centres=cluster_centres)
            target1= self.get_target_distb(cluster_probs1)
            cluster_loss_1 = self.kl_div((cluster_probs1+1e-08).log(), target1)/cluster_probs1.shape[0]

            cluster_probs2= self.get_cluster_prob(self, hidden2, cluster_centres=cluster_centres)
            target2= self.get_target_distb(cluster_probs2)
            cluster_loss_2 = self.kl_div((cluster_probs2+1e-08).log(), target2)/cluster_probs2.shape[0]

            return (cluster_loss_1+cluster_loss_2)/2.0
        




