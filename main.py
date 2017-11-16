import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.cluster import KMeans
from helper import *
from model import *

use_cuda = torch.cuda.is_available()

SOS_token = 1
EOS_token = 0
UNK_token = 2

MAX_SEQ_LEN = 20

def translate(words, dict, n_words):
    res = []
    for word in words:
        try:
            if int(dict[word]) > n_words:
                w = "UNK"
            else:
                w = word
        except:
            w = "UNK"
        res.append(int(dict[w]))
    return res

def translate_back(ints, dict):
    res = []
    for i in ints:
        res.append(dict[i])
    return res

if __name__ == '__main__':

    # n_words = 300003
    n_words = 10000 # for testing
    embedded_size = 256
    rnn_size = 1024
    num_class = 10 # number of cluster classes
    
    test_dict = json.loads(open("data/vocab.json", "r").readline())
    reverse_test_dict = dict(zip(test_dict.values(), test_dict.keys()))
    test_prob_dict = json.loads(open("data/vocab_prob.json", "r").readline())

    test_ans_file = open("data/valid.txt", "r")
    test_ans = []
    test_ans_prob = []
    test_p = []
    test_p_prob = []
    ans_samples = []
    max_ans_length = 0
    max_p_length = 0
    for i in range(30):
        ans_samples.append(test_ans_file.readline())
        ans_samples[-1] = ans_samples[-1].translate(string.maketrans("", ""), string.punctuation)
        ans_samples[-1] = ans_samples[-1].strip("\r\n").split("\t")
        ans_samples[-1][0] = ans_samples[-1][0].split(" ")
        ans_samples[-1][1] = ans_samples[-1][1].split(" ")
        max_ans_length = max(max_ans_length, len(ans_samples[-1][1]))
        max_p_length = max(max_p_length, len(ans_samples[-1][0]))
    for i in range(30):
        test_ans.append([])
        test_ans_prob.append([])
        test_p.append([])
        test_p_prob.append([])
        ans_line = ans_samples[i]
        j = 0
        k = 0
        for word in ans_line[1]:
            try:
                if int(test_dict[word]) > n_words:
                    w = "UNK"
                else:
                    w = word
            except:
                w = "UNK"
            test_ans[-1].append(int(test_dict[w]))
            test_ans_prob[-1].append(float(test_prob_dict[w]))
            j += 1
        for word in ans_line[0]:
            try:
                if int(test_dict[word]) > n_words:
                    w = "UNK"
                else:
                    w = word
            except:
                w = "UNK"
            test_p[-1].append(int(test_dict[w]))
            test_p_prob[-1].append(float(test_prob_dict[w]))
            k += 1
        if j < max_ans_length:
            for j in range(j, max_ans_length):
                test_ans[-1].append(int(test_dict["<EOS>"]))
                test_ans_prob[-1].append(float(test_prob_dict["<EOS>"]))
        if k < max_p_length:
            for _ in range(k, max_p_length):
                test_p[-1].append(int(test_dict["<EOS>"]))
                test_p_prob[-1].append(float(test_prob_dict["<EOS>"]))
    test_ans = torch.LongTensor(test_ans)
    # test_ans_onehot = torch.zeros(30, 63, embedded_size).scatter_(1, test_ans.data, 1)
    # print(test_ans_onehot)
    test_ans_prob = Variable(torch.FloatTensor(test_ans_prob))
    test_p = Variable(torch.LongTensor(test_p))
    test_p_prob = Variable(torch.FloatTensor(test_p_prob))
    embedder = nn.Embedding(n_words, embedded_size, padding_idx=EOS_token)
    embedded_answers = embedder(Variable(test_ans))
    embedded_problems = embedder(test_p)
    if (use_cuda):
        embedded_answers = embedded_answers.cuda()
        test_ans_prob = test_ans_prob.cuda()
        embedded_problems = embedded_problems.cuda()
        test_p_prob = test_p_prob.cuda()
        test_ans = test_ans.cuda()
    sent_vec = sentence_vector_wr_pca(embedded_answers, test_ans_prob, max_ans_length, 1e-4) 
    
    km = KMeans()
    if (use_cuda):
        km.fit(sent_vec.cpu().data.numpy())
        target_C_labels = torch.from_numpy(km.labels_).type(torch.LongTensor).cuda()
    else:
        km.fit(sent_vec.data.numpy())
        target_C_labels = torch.from_numpy(km.labels_).type(torch.LongTensor)

    encoder = Encoder(embedded_size, rnn_size)
    decoder = Decoder(embedded_size, rnn_size, n_words, EOS_token)
    classifier = Classifier(rnn_size, num_class)
    CE_loss = torch.nn.CrossEntropyLoss()
    optim_vars = list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(optim_vars, lr=1e-3)
    val_question = ["who", "are", "you", "<EOS>"]
    val_question_int = Variable(torch.LongTensor([translate(val_question, test_dict, n_words)]))
    embedded_val = embedder(val_question_int)
    if use_cuda: embedded_val = embedded_val.cuda()
    test_softmax = torch.nn.Softmax()

    for epoch in range(10000):
        optimizer.zero_grad()
        # print(decoder.linear.bias.grad)
        output, hidden = encoder(embedded_problems)
        C, score = classifier(torch.squeeze(hidden, 0))
        init_C = torch.unsqueeze(C, 1)
        res_codes = decoder(init_C, hidden, max_ans_length, True, embedded_answers)
        loss = CE_loss(score, Variable(target_C_labels, requires_grad=False))
        for i in range(max_ans_length):
            loss += CE_loss(res_codes[:, i, :], Variable(test_ans[:, i], requires_grad=False))
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 100 == 0:
            print(loss)
            output, hidden = encoder(embedded_problems)
            C, score = classifier(torch.squeeze(hidden, 0))
            init_C = torch.unsqueeze(C, 1)
            r = torch.squeeze(decoder(init_C, hidden, max_ans_length), 0)
            for i in range(30):
                _, res = torch.max(test_softmax(r[i]), 1)
                print(translate_back(res.data, reverse_test_dict))
                _, res = torch.max(test_softmax(res_codes[i]), 1)
                print(translate_back(res.data, reverse_test_dict))


        
