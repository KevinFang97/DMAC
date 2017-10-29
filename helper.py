#helper functions
import matplotlib.pyplot as plt #using PCA here
import numpy as np
import torch
from torch.autograd import Variable


#d1vec is a 1 dim vec i.e. (batch_size,)
#should be rewrite so that d1vec is Variable
def one_hot(d1vec,voca_size):
  batch_size = d1vec.shape[0]
  temp = np.zeros((batch_size,voca_size))
  temp[xrange(batch_size), d1vec] = 1
  return temp


#wr step for computing sentence vector
#ref page4 of paper about WR+PCA
#probability is a list of probabilities of words in sentence

#answers_prob shape: (N,W)
#embedded_answers shape: (N,W,D)
def sentence_vector_wr_pca(answers, embedder, answers_prob, answers_length, a):
  embedded_answers = embedder(answers)
  if torch.cuda.is_available():
    embedded_answers = embedded_answers.cuda()
    answers_prob = answers_prob.cuda()
  N, W, _ = embedded_answers.size()
  answers_prob = answers_prob.view([N, W, 1])
  #doing wr
  embedded_answers = (a / (a + answers_prob)) * embedded_answers #shape: (N,W,D)
  #here we also count <SOS>,<EOS>,<UNK> in calculation of wr
  sentence_vec = torch.mean(embedded_answers, dim=1) #### Here not strictly follow the original paper
  # print(sentence_vec[0,0])
  # print(sentence_vec[0,1])
  #Now do pca
  sentence_vec_norm = sentence_vec - torch.mean(sentence_vec, 1, keepdim=True)
  sigma = torch.matmul(sentence_vec_norm, torch.t(sentence_vec_norm))
  sigma /= sentence_vec_norm.size()[1]
  u, _, _ = torch.svd(sigma.data)
  u = Variable(torch.unsqueeze(u[:, 0], 0))
  uutv = torch.matmul(torch.t(u).mm(u), sentence_vec)
  return sentence_vec - uutv
