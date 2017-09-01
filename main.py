
import torch
import torch.nn as nn





use_cuda = torch.cuda.is_available

SOS_token = 1
EOS_tokem = 0
UNK_token = 2

MAX_SEQ_LEN = 20

#-------------Train The Model------------#

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

	