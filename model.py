import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

SOS_token = 1
EOS_token = 0
UNK_token = 2

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
  def __init__(self, input_size, hidden_size, n_layers=1, use_cuda=torch.cuda.is_available()):
    super(Encoder, self).__init__()
    # self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda
    self.gru = nn.GRU(input_size, hidden_size, n_layers,
                      dropout=0.2, batch_first=True)
    if (use_cuda):
      self.cuda()

  def forward(self, input, hidden=None):
    if hidden is None:
      hidden = Variable(torch.randn(1, input.size()[0], self.hidden_size))
      if (self.use_cuda):
        hidden = hidden.cuda()
    output, hidden = self.gru(input, hidden)
    return output, hidden


#decode a sentence using hidden as given and input as <SOS>
#input a batch: (batch_size, voca_size) (one-hot)
#input hidden: (batch_size, hidden_size) (prev output hidden)
#output a batch: (batch_size, voca_size) (softmaxed)
#output hidden: (batch_size, hidden_size)

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_word, EOS_token, n_layers=1, use_cuda=torch.cuda.is_available()):
    super(Decoder, self).__init__()
    # self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda
    self.input_size = input_size
    self.EOS_token = EOS_token
    self.gru = nn.GRU(input_size, hidden_size, n_layers,
                      dropout=0.2, batch_first=True)
    self.linear_C = nn.Linear(1, input_size)
    self.linear = nn.Linear(hidden_size, input_size)
    self.linear_result = nn.Linear(input_size, num_word)
    self.relu = nn.ReLU()
    if (use_cuda):
      self.cuda()

  def forward(self, C, init_hidden, embedder, max_len=30):
    result = None
    hidden = init_hidden
    output = self.linear_C(C)
    for i in range(max_len):
      output, hidden = self.step(output, hidden, embedder)
      if result is None:
        result = output
      else:
        result = torch.cat([result, output], 1)
    result = self.relu(self.linear_result(result))
    return result

  def step(self, input, hidden, embedder):
    output, hidden = self.gru(input, hidden)
    hidden.detach_()
    output = torch.squeeze(output)
    output = self.linear(output)
    output = torch.unsqueeze(output, 1)
    return output, hidden
      

class Classifier(nn.Module):
  def __init__(self, rnn_size, num_class, use_cuda=torch.cuda.is_available()):
    super(Classifier, self).__init__()
    self.rnn_size = rnn_size
    self.num_class = num_class
    self.use_cuda = use_cuda
    self.softmax = torch.nn.Softmax()
    self.classifier = torch.nn.Linear(rnn_size, num_class)
    if use_cuda:
      self.cuda()
  
  def forward(self, input):
    score = self.classifier(input)
    _, C = torch.max(self.softmax(score), -1)
    C = torch.unsqueeze(C, 1)
    if self.use_cuda:
      C = C.type(torch.cuda.FloatTensor)
    else:
      C = C.type(torch.FloatTensor)
    return C, score
    