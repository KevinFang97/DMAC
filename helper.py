#helper functions
import matplotlib.pyplot as plt #using PCA here
import numpy as np

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

'''
########OLD VERSION FULL OF BUGS########
def sentence_vector_wr(embedded_sentence, prob, sentence_size, parameter_a):
	sentence_vec = np.zeros(embedding_size)
  for i in range(sentence_size):
    word_vec_ranked = (parameter_a / (parameter_a + prob[i+1])) * embedded_sentence[i]
		sentence_vec += word_vec_ranked
  sentence_vec = sentence_vec / sentence_size
  return sentence_vec
'''
def sentence_vector_wr_vectorize(answers, embedder, answers_prob, answers_length, a):
  #answers_prob shape: (N,W)
  #embedded_answers shape: (N,W,D)
  embedded_answers = embedder(answers)
  N, W, D = embedded_answers.shape
  answers_prob = answers_prob.reshape((N,W,1))
  #doing wr
  embedded_answers = (a/(a+answers_prob))*embedded_answers #shape: (N,W,D)
  #here we also count <SOS>,<EOS>,<UNK> in calculation of wr
  sentence_vec = np.mean(embedded_answers, axis = 1)
  return sentence_vec
  
######DONT KNOW HOW TO COMPUTE PCA######
def sentence_vector_pca(sentence_vector_list):
  list_length = len(sentence_vector_list)
  sentence_vector_array = torch.FloatTensor(sentence_vector_list)
  sigma = torch.matmul(torch.transpose(sentence_vector_array, 1, 0), sentence_vector_array)
  sigma = sigma / list_length
  u, _, _ = torch.svd(sigma)
  u = u[:, 0]
  uut = torch.matmul(u.T, u)
  sentence_vector_array = sentence_vector_array * (1 - uut)
  return sentence_vector_array

'''
def cluster_pred(answer, cluster_center, num_cluster):
  #answer size: (batch_size, max_length)

'''