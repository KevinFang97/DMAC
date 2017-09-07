#helper functions
import matplotlib.pyplot as plt #using PCA here
import numpy as np

#wr step for computing sentence vector
#ref page4 of paper about WR+PCA
#probability is a list of probabilities of words in sentence
######<SOS>, <EOS> not included######
def sentence_vector_wr(embedded_sentence, prob, sentence_size, parameter_a):
	sentence_vec = np.zeros(embedding_size)
  for i in range(sentence_size):
    word_vec_ranked = (parameter_a / (parameter_a + prob[i+1])) * embedded_sentence[i]
		sentence_vec += word_vec_ranked
  sentence_vec = sentence_vec / sentence_size
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
