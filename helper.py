#helper functions

#wr step for computing sentence vector
#ref page4 of paper about WR+PCA
#probability is a list of probabilities of words in sentence
######not clear about whether EOS is computed here######
def sentence_vector_wr(embedded_sentence, probability, sentence_size, parameter_a):
  result = 0.0
  for i in range(sentence_size):
    result += (parameter_a / (parameter_a + probability[i])) * embedded_sentence[i]
  result = result / sentence_size
  return result
  
