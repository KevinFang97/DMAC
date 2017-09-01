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
  iii)for each rnn cell: input = prev_output + coa, 
                         output = word_one_hot or word_embedded? (not clear about part(iii), TO BE MODIFIED)
3.loss func
