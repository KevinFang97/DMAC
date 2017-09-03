import torch
import torch.nn as nn
import torch.nn.functional as F

###class(cluster) prediction###

#input: sentence vector, size = embedding_size
#output: one-hot vector, size = num_of_cluster
class Classifier(nn.Module):
  def __init__(self, embedding_size, num_of_cluster):
    super(Classifier, self).__init__()
    self.num_of_cluster = num_of_cluster
    self.fc1 = nn.Linear(embedding_size, embedding_size/2)
    self.fc2 = nn.Linear(embedding_size/2, num_of_cluster)
    
  def forward(self, input):
    input = F.relu(self.fc1(input))
    input = F.softmax(self.fc2(input))


###answer prediction###
class Encoder(nn.Module):
  def __init__(self, input_size, n_layers=1):
    super(Encoder, self).__init__()
    self.n_layers = n_layers
    
    self.gru = nn.GRU
 
