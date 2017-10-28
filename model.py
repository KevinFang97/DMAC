import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###class(cluster) prediction###
#now using kmeans but NN
#input: sentence vector, size = embedding_size
#output: one-hot vector, size = num_of_cluster
#class Classifier(nn.Module):
#  def __init__(self, embedding_size, num_of_cluster):
#    super(Classifier, self).__init__()
#    self.num_of_cluster = num_of_cluster
#    self.fc1 = nn.Linear(embedding_size, embedding_size/2)
#    self.fc2 = nn.Linear(embedding_size/2, num_of_cluster)
#
#  def forward(self, input):
#    input = F.relu(self.fc1(input))
#    input = F.softmax(self.fc2(input))


###answer prediction###


##########Encoder & Decoder##############
#encode a batch of sentence input: input shape: (batch_size, sentence_size, embedding_size)
class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, n_layers=1):
    super(Encoder, self).__init__()
    # self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=0.2, batch_first=True)

  def forward(self, input, hidden):
    '''
    for i in range(self.n_layers):
      output, hidden = self.gru(output, hidden)
    '''
    output, hidden = self.gru(input, hidden)
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if use_cuda:
      return result.cuda()
    else:
      return result

#decode a sentence using hidden as given and input as <SOS>
#input a batch: (batch_size, voca_size) (one-hot)
#input hidden: (batch_size, hidden_size) (prev output hidden)
#output a batch: (batch_size, voca_size) (softmaxed)
#output hidden: (batch_size, hidden_size)


class Decoder(nn.Module):
  def __init__(self, batch_size, hidden_size, voca_size, n_layers=1):
    super(Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.embedding = nn.Embedding(voca_size, hidden_size)
    self.gru = nn.GRU(hidden_size, voca_size, dropout=0.2, batch_first=True)
    for w in self.gru.parameters():  # initialize the gate weights with orthogonal
      if w.dim() > 1:
        weight_init.orthogonal(w)
    self.out = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(batch_size, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
    # output[0] shape: (batch_size, hidden_size)
    output = self.softmax(self.out(output[0]))
    #output shape: (batch_size, voca_size)
    return output, hidden

#  def initHidden(self):
#    result = Variable(torch.zeros(1, 1, self.hidden_size))
#    if use_cuda:
#      return result.cuda()
#    else:
#      return result

##########Encoder & Decoder##############
#


'''
class Classifier(nn.Module):
  def __init__(self, cluster_centers, num_of_cluster):
    super(Classifier, self).__init__()
    self.cluster_centers = cluster_centers
    self.num_of_cluster = num_of_cluster

  def forward(answer, batch_size):
    #answer size: (batch_size, max_seq_len, embedding_size)
    #cluster_centers size: (num_cluster, embedding_size)
    #result size: (batch_size, num_cluster)
    #using loop which is more readable, less error prone, maybe matrix operation method will be addded
'''