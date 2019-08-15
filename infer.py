from keras.models import Model,load_model
from keras.layers import Input,LSTM,Dense
import numpy as np
import pandas as pd

from data_util import get_vocab_dict
from data_util import get_xy_data
from data_util import get_rev_dict
import data_util

latent_dim = 256

# 语料向量化
input_texts = []
target_texts = []
input_token_index = []
target_token_index = []

def seq2vec(input_seq):
    vec_list = []
    for word in input_seq:
        vec_list.append(input_token_index[word])
    return vec_list

# 开始inference
def decoder_sequence(input_seq):
     # Encode the input as state vectors
     states_value = encoder_model.predict(input_seq)

     target_seq = np.zeros((1,1,data_util.num_decoder_tokens))
     # '\t' is starting character
     target_seq[0,0,target_token_index['\t']] = 1

     # Sampling loop for a batch of sequences
     stop_condition = False
     decoded_sentence= ''
     while not stop_condition:
          output_tokens, h, c = decoder_model.predict(
               [target_seq] + states_value)
          sampled_token_index = np.argmax(output_tokens[0,-1,:])
          sampled_char = reverse_target_char_index[sampled_token_index]
          decoded_sentence += sampled_char
          if(sampled_char == '\n' or len(decoded_sentence) > data_util.max_decoder_seq_length):
               stop_condition = True

          # Update the target sequenco to predict next token
          target_seq = np.zeros((1,1,data_util.num_decoder_tokens))
          target_seq[0,0,sampled_token_index] = 1

          # Update state
          states_value = [h, c]

     return decoded_sentence

def predict_ans(question):
     input_seq = np.zeros((1, data_util.max_encoder_seq_length, data_util.num_encoder_tokens),dtype='float16')
#    input_seq = seq2vec(question)
     for t, char in list(enumerate(question)):
          input_seq[0,t,input_token_index[char]] = 1
     decoded_sentence = decoder_sequence(input_seq)
     return decoded_sentence

if __name__ == "__main__":
     input_texts, target_texts = get_xy_data()
     input_token_index, target_token_index = get_vocab_dict(input_texts, target_texts)
     reverse_input_char_index, reverse_target_char_index = get_rev_dict(input_token_index, target_token_index)
     encoder_model = load_model('encoder_model.h5')
     decoder_model = load_model('decoder_model.h5')
     print('Decoded sentence:', predict_ans('我要升级了'))