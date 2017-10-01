import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.cluster import KMeans


use_cuda = torch.cuda.is_available

SOS_token = 1
EOS_tokem = 0
UNK_token = 2

MAX_SEQ_LEN = 20

#use list represent list, np.array represent normal vector, pytorch.Variable represent vector/tensor in NN

#-------------Cluster Training------------#

'''
############OLD VERSION FULL OF BUGS##############
#probability dict should be calculated while generating word dict
#here the input of embedder is a sentence(int value,  padded), of course we need to make all embedder the SAME one
def cluster_train(answers, embedder, answers_length, prob_dict, num_cluster, parameter_a):
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
	kmeans = KMeans(n_clusters=num_cluster, n_jobs=-1)
	centers = kmeans.cluster_centers_ #shape: [num_cluster, embedding_size]
	
	#return the kmean center
	return centers
'''

def cluster_train_vectorize(answers, embedder, answers_length, prob_dict, num_cluster, parameter_a):
  #answers shape: (Number_of_answers, sentence_size)
  #answers_length shape: (Number_of_answers, )
  N,W = answers.shape
	answers_prob = np.zeros((answers.shape)) #the list of sentence vectors of answers
  
  #translate answers in to its prob
  for word in prob_dict:
    answers_prob[answer == word] = prob_dict[word] 
  
  #sentence_vec shape: (Number_of_answers,embedding_size)
  #doing wr
  sentence_vec = sentence_vec_wr_vectorize(answers, embedder, answers_prob, answers_length, parameter_a)
  #doing PCA
  sentence_vec = sentence_vector_pca_vectorize(sentence_vec)
    
  #clustering using answer_vec, use k-means
	kmeans = kMeans(n_clusters=num_cluster, n_jobs=-1).fit(sentence_vec)	
	#return the kmean center and labels
  #center shape: (num_cluster, embedding_size)
  #label shape: (Number_of_answers, )
	return kmeans.cluster_centers_, kmeans.labels_
  
  
  
#-------------End of Cluster Training-----------#


#-------------Train The Main Model------------#

#(Development use)List the models needed to be implemented in model.py, which are used here
#1.Encoder
#2.Decoder
#...... =.= seems no more

def _train_step(q_batch, a_batch, q_lens, a_lens, classifier, embedder, encoder, decoder, classifier_optimizer, embedder_optimizer, encoder_optimizer, decoder_optimizer, voca_size):
	'''
  Train one instance
  '''

	#zero-grad for optimizers 
	classifier_optimizer.zero_grad()
  embedder_optimizer.zero_grad()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

	batch_size = len(q_batch)
  answer_size = a.size(1) #num of words in an answer, incase size of q and a are not the same

  #a shape, q shape: (batch_size, sentence_size)
	q = Variable(q_batch)
	q = q.cuda() if use_cuda else q
	a = Variable(a_batch)
	a = a.cuda() if use_cuda else a

	#embed q, q_emb shape: (batch_size, sentence_size, embedding_size)
	q_emb = embedder(q)
	#encode q
  _, q_enc = encoder(q_emb) #q_enc shape: (batch_size, encoder_hidden_size)
	#cluster a
	loss_clu, a_clu = classifier(a) #a_clu size: (batch_size, num_cluster), each cluster vec is like (0.12,0.67,0.21) where each number represents the prob in this cluster
  
  decoder_input = np.full((batch_size,1), SOS_token)
  decoder_input = one_hot(decoder_input)
  decoder_input = Variable(torch.from_numpy(decoder_input))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input # [batch_sz x voca_size]
  #decoder_hidden shape: [1 x batch_sz x decoder_hidden_size] (1=n_layers) 
  #decoder_hidden_size = encoder_hidden_size + num_cluster
  decoder_hidden = torch.cat([q_enc,a_clu],1).unsqueeze(0) 
  use_teach_force = True if random.random() < p_teach_force else False
  out = Variable(torch.from_numpy(np.zeros((batch_size,answer_size))))
  decoder_loss = 0
  #decoding process
	for di in range(answer_size):
    #decoder_output: [batch_sz x voca_sz]
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    #should be rewrite in to pytorch instead of nunmpy
    decode_label = np.argmax(decoder_output, axis=1) #shape: (batch_size,)
    out[:,di] = decode_label
    decoder_loss += -np.mean(np.log(soft_y[xrange(batch_size),decode_label]))
    #TO-DO: revise here, cuz our input need to be one-hot vec, since output is only softmaxed, we need to
    #       1)predicted
    #       2)compute loss
    #“Teacher forcing” is the concept of using the real target outputs as each next input, 
    #instead of using the decoder’s guess as the next input. 
    #Using teacher forcing causes it to converge faster 
    #but when the trained network is exploited, it may exhibit instability.
    if use_teach_force:
      decoder_input = a[:,di].unsqueeze(1)  # Teacher forcing
            #print("decoder_input_sz_1:")
            #print(decoder_input.size())
    else:
      topi = decoder_output[:,-1].max(1)[1] # topi:[batch_sz x 1] indexes of predicted words
      decoder_input = topi#Variable(torch.LongTensor(ni))
            #ni = topi.cpu().numpy().squeeze().tolist() #!!
            #if ni == EOS_token:
            #    break
    decoder_input=embedder(decoder_input)
    #print(out)