#note:sentence_size here does not include <SOS> or <EOS>
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


use_cuda = torch.cuda.is_available

SOS_token = 1
EOS_tokem = 0
UNK_token = 2

MAX_SEQ_LEN = 20

#use list represent list, np.array represent normal vector, pytorch.Variable represent vector/tensor in NN

#-------------Cluster Training------------#

#probability dict should be calculated while generating word dict
#here the input of embedder is a sentence(int value,  padded), of course we need to make all embedder the SAME one
def cluster_train(answers, embedder, answers_length, prob_dict, num_cluster):
	a = 1 #the Parameter in wr calculation, value TO BE DECIDED
	answers_vec = [] #the list of sentence vectors of answers
	
	#doing wr+PCA
	#wr
	for i in range(answers):
		prob = [] #prob of words in this sentence
		for j in range(len(answers[i])):
			word_prob = prob_dict[answers[i][j+1]] #the ith sentence, j+1th word in it
			prob.append(word_prob) #finish retrieving the prob of word in this sentence
		sentence_vec_wr = sentence_vector_wr(embedder(answers[i]), prob, answers_length[i], a) #wr is the sentence_vec (not PCAed yet)
		answers_vec.append(sentence_vec_wr)
	#doing PCA for all sentence_vec of answers
	sentence_vector_pca(answers_vec)

	#clustering using answer_vec, use k-means
	kmeans = kMeans(n_clusters=num_cluster, n_jobs=-1)
	centers = kmeans.cluster_centers_ #shape: [num_cluster, embedding_size]
	
	#return the kmean center
	return centers
	
#-------------End of Cluster Training-----------#


#-------------Train The Main Model------------#

#(Development use)List the models needed to be implemented in model.py, which are used here
#1.Encoder
#2.Decoder
#...... =.= seems no more

def _train_step(q_batch, a_batch, q_lens, a_lens, classifier, embedder, encoder, decoder, classifier_optimizer, embedder_optimizer, encoder_optimizer, decoder_optimizer):
	'''
    Train one instance
    '''

	#zero-grad for optimizers 
	classifier_optimizer.zero_grad()
    embedder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

	batch_size = len(q_batch)

	q = Variable(q_batch)
	q = q.cuda() if use_cuda else q
	a = Variable(a_batch)
	a = a.cuda() if use_cuda else a

	#embed q
	q_emb = embedder(q)
	#encode q
    _, q_enc = encoder(q_emb) #q_enc size: TO BE ADDED
	#cluster a
	loss_clu, a_clu = classifier(a, a_lens) #a_clu size: (batch_size, num_cluster), each cluster vec is like (0.12,0.67,0.21) where each number represents the prob in this cluster

	qnc = torch.cat([q_enc,a_clu],1) #HOW DOES IT LOOK LIKE AFTER CAT??? NEED EXPERIMENT!!! WHAT AXIS?？？
	#lets assume last line is correct

	decoder_input = qnc #NEED MODIFIED

	#dont know how to write decoder in pytorch yet, TBC