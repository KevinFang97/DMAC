# DMAC
Dialogue Model with Answer Clustering

V1.0:
1.use random embedding
2.use wr + pca in answer clustering #ref: A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
3.use hyperparameters (for dimension/#clusering/type_of_cell/para_in_loss_function)?


Main Procedure:
1.NN for answer clustering:
  i)for each answer, use wr+pca to compute its sentence vector
  ii)train the nn whose input is sentence vector and output is class(clustering). (simple unsupervised, classification problem) (use SVM or not?)
  iii)save the model after training
2.NN for answer/class prediction:
  i)encode question (question -> question_embedding (qe) )
  ii)NN to predict class (of the answer) (easy) (qe->coa (class_of_answer))
	iii)rnn_input = encoder_output + coa (in probability?)
  iv)for each rnn cell: input = prev_output (or <SOS>) (one_hot_vec), 
                         output = probability_vec (size = vocabulary_size) (not normalized)
                         output = softmax(output)
                         loss = f(output,real_answer), output = maxarg(output)
3.loss func:
  loss_in_cluster_pred + parameter*loss_in_answer_pred


#using kmeans++ to cluster, output will be n center point. (n = num_of_cluster if using kmeans)
#while using the cluster model, for each point, clustering result should be a normalized vector representing probability (calculated using euclidean distance, transfer to represent probability using its reciprocal, normalize using softmax)