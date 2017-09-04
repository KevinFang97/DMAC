#note:sentence_size here does not include <SOS> or <EOS>
import torch
import torch.nn as nn





use_cuda = torch.cuda.is_available

SOS_token = 1
EOS_tokem = 0
UNK_token = 2

MAX_SEQ_LEN = 20

#-------------Train The Cluster Model------------#
def cluster_train_step(q_batch)

#probability dict should be calculated while generating word dict
#here the input of embedder is a sentence(int value,  padded)
def cluster_train(answers, embedder, answer_length, prob_dict):
	answer_vec = []
	a = 1 #TO BE DECIDED

	
	
	for i in range(answers):
		prob = [] #prob of the sentence
		for j in range(len(answers[i])):
			word_prob = prob_dict[answers[i][j+1]]
			prob.append(word_prob) #finish retrieving the prob of word in this sentence
		wr = sentence_vector_wr(embedder(answers[i]), prob, answers_length[i], a) #wr is the sentence_vec (not PCAed yet)
		answer_vec.append(wr)
		
	sentence_vector_pca(answer_vec) #doing PCA for all sentence_vec of answers
	
	#clustering using answer_vec, ALGO TO BE DECIDED
	
	
#-------------End of The Cluster Model-----------#


#-------------Train The Main Model------------#

#(Development use)List the models needed to be implemented in model.py, which are used here
#1.

def _train_step(q_batch, a_batch, q_lens, a_lens, embedder, ):
	
	#zero-grad for optimizers 
	#TBA

	batch_size = len(q_batch)

	q = Variable(q_batch)
	q = q.cuda() if use_cuda else q
	a = Variable(a_batch)
	a = a.cuda() if use_cuda else a

	q_emb = embedder(q)
	a_emb = embedder(a)

	
