import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.cluster import KMeans
from helper import *
from model import *

use_cuda = torch.cuda.is_available

SOS_token = 1
EOS_token = 0
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
    sentence_vec_wr = sentence_vector_wr(embedder(
        answers[i]), prob, answers_length[i], a) #wr is the sentence_vec (not PCAed yet)
    answers_vec.append(sentence_vec_wr)
  #doing PCA for all sentence_vec of answers
  sentence_vector_pca(answers_vec)

  #clustering using answer_vec, use k-means
  kmeans = KMeans(n_clusters=num_cluster, n_jobs=-1)
  centers = kmeans.cluster_centers_ #shape: [num_cluster, embedding_size]

  #return the kmean center
  return centers
'''


def cluster_train_vectorize(answers, embedder, answers_length, prob_dict,
                            num_cluster, parameter_a):
    #answers shape: (Number_of_answers, sentence_size)
    #answers_length shape: (Number_of_answers, )
    N, W = answers.shape
    # the list of sentence vectors of answers
    answers_prob = np.zeros((answers.shape))

    #translate answers in to its prob
    for word in prob_dict:
        answers_prob[answer == word] = prob_dict[word]

    #sentence_vec shape: (Number_of_answers,embedding_size)
    #doing wr
    sentence_vec = sentence_vec_wr_vectorize(answers, embedder, answers_prob,
                                             answers_length, parameter_a)
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


def _train_step(q_batch, a_batch, q_lens, a_lens, classifier, embedder,
                encoder, decoder, classifier_optimizer, embedder_optimizer,
                encoder_optimizer, decoder_optimizer, voca_size, clu_loss_reg):
    '''
  Train one instance

  #zero-grad for optimizers
  classifier_optimizer.zero_grad()
  embedder_optimizer.zero_grad()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
'''

    batch_size = len(q_batch)
    answer_size = a.size(
        1)  #num of words in an answer, incase size of q and a are not the same

    #a shape, q shape: (batch_size, sentence_size)
    q = Variable(q_batch)
    q = q.cuda() if use_cuda else q
    a = Variable(a_batch)
    a = a.cuda() if use_cuda else a

    #embed q, q_emb shape: (batch_size, sentence_size, embedding_size)
    q_emb = embedder(q)
    #encode q
    _, q_enc = encoder(q_emb)  #q_enc shape: (batch_size, encoder_hidden_size)
    #cluster a
    clu_loss, a_clu = classifier(
        a
    )  #a_clu size: (batch_size, num_cluster), each cluster vec is like (0.12,0.67,0.21) where each number represents the prob in this cluster

    decoder_input = np.full((batch_size, 1), SOS_token)
    decoder_input = Variable(torch.from_numpy(decoder_input))
    decoder_input = one_hot(decoder_input, voca_size)
    decoder_input = decoder_input.cuda(
    ) if use_cuda else decoder_input  # [batch_sz x voca_size]
    #decoder_hidden shape: [1 x batch_sz x decoder_hidden_size] (1=n_layers)
    #decoder_hidden_size = encoder_hidden_size + num_cluster
    decoder_hidden = torch.cat([q_enc, a_clu], 1).unsqueeze(0)
    use_teach_force = True if random.random() < p_teach_force else False
    predict = Variable(torch.from_numpy(np.zeros((batch_size, answer_size))))
    decoder_loss = 0
    #decoding process
    for di in range(answer_size):
        #decoder_output: [batch_sz x voca_sz]
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        #should be rewrite in to pytorch instead of nunmpy
        decode_label = np.argmax(decoder_output, axis=1)  #shape: (batch_size,)
        predict[:, di] = decode_label
        decoder_loss += -np.mean(
            np.log(soft_y[xrange(batch_size), decode_label]))
        #TO-DO: revise here, cuz our input need to be one-hot vec, since output is only softmaxed, we need to
        #       1)predicted
        #       2)compute loss
        #Teacher forcing is the concept of using the real target outputs as each next input,
        #instead of using the decoders guess as the next input.
        #Using teacher forcing causes it to converge faster
        #but when the trained network is exploited, it may exhibit instability.
        if use_teach_force:
            decoder_input = a[:, di].unsqueeze(1)  #size: [batch_size,]
            decoder_input = one_hot(decoder_input,
                                    voca_size)  #size: [batch_size,voca_size]

        else:
            decoder_input = one_hot(decode_label,
                                    voca_size)  #size: [batch_size, voca_size]

    loss = decoder_loss + clu_loss_reg * clu_loss
    loss.backward()
    '''
  embedder_optimizer.step()
  encoder_optimizer.step()
  decoder_optimizer.step()
  classifier_optimizer.step()
  '''

    #return type need to be modified
    return loss.data[0], kl_loss.data[0], decoder_loss.data[0]
    '''
def train(embedder, encoder, hidvar, decoder, data_loader, vocab, n_iters, p_teach_force=0.5, model_dir,
          save_every=5000, sample_every=100, print_every=10, plot_every=100, learning_rate=0.00005):
'''


if __name__ == '__main__':


    n_words = 300003
    # n_words = 20 # for testing
    embedded_size = 256
    rnn_size = 1024
    
    test_dict = json.loads(open("data/vocab.json", "r").readline())
    # test_dict = dict(zip(test_dict.values(), test_dict.keys()))
    test_prob_dict = json.loads(open("data/vocab_prob.json", "r").readline())

    test_ans_file = open("data/valid.txt", "r")
    test_ans = []
    test_ans_prob = []
    ans_samples = []
    max_ans_length = 0
    for _ in range(30):
        ans_samples.append(test_ans_file.readline())
        ans_samples[-1] = ans_samples[-1].translate(string.maketrans("", ""), string.punctuation)
        ans_samples[-1] = ans_samples[-1].strip("\r\n").split("\t")
        max_ans_length = max(max_ans_length, len(ans_samples[-1][1]))
    for i in range(30):
        test_ans.append([])
        test_ans_prob.append([])
        ans_line = ans_samples[i]
        j = 0
        for word in ans_line[1]:
            try:
                test_ans[-1].append(int(test_dict[word]))
                test_ans_prob[-1].append(float(test_prob_dict[word]))
            except:
                test_ans[-1].append(int(test_dict["UNK"]))
                test_ans_prob[-1].append(float(test_prob_dict["UNK"]))
            j += 1
        if j < max_ans_length:
            for j in range(j, max_ans_length):
                test_ans[-1].append(int(test_dict["UNK"]))
                test_ans_prob[-1].append(float(test_prob_dict["UNK"]))
    test_ans = Variable(torch.LongTensor(test_ans))
    test_ans_prob = Variable(torch.FloatTensor(test_ans_prob))
    
    embedder = nn.Embedding(n_words, embedded_size, padding_idx=EOS_token)
    sent_vec = sentence_vector_wr_vectorize(test_ans, embedder, test_ans_prob, max_ans_length, 1e-4) 
    encoder = Encoder(embedded_size, rnn_size)
    #for testing
