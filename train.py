from keras.models import Model
from keras.layers import Input,LSTM,Dense
import numpy as np
import pandas as pd
import data_util
from data_util import get_vocab_dict
from data_util import get_xy_data

# 定义超参数
batch_size = 32
epochs = 100
latent_dim = 256

input_texts = []
target_texts = []
input_token_index = []
target_token_index = []
encoder_input_data = None
decoder_input_data = None
decoder_target_data = None

def data_deal():
     global encoder_input_data,decoder_input_data,decoder_target_data
     global input_texts, target_texts, input_token_index,target_token_index
     input_texts, target_texts = get_xy_data()
     input_token_index, target_token_index = get_vocab_dict(input_texts, target_texts)

     # 每个input_text句子都是一个二维矩阵，
     # 那么input_texts是多个二维矩阵组合的三维矩阵
     encoder_input_data = np.zeros(
          (len(input_texts), data_util.max_encoder_seq_length, len(input_token_index)),dtype='float32')
     decoder_input_data = np.zeros(
          (len(input_texts), data_util.max_decoder_seq_length, len(target_token_index)),dtype='float32')
     decoder_target_data = np.zeros(
          (len(input_texts), data_util.max_decoder_seq_length, len(target_token_index)),dtype='float32')

     for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
          for t, char in enumerate(input_text):
               encoder_input_data[i, t, input_token_index[char]] = 1
          for t, char in enumerate(target_text):
               decoder_input_data[i, t, target_token_index[char]] = 1
               if t > 0:
                    decoder_target_data[i, t-1, target_token_index[char]] =1


def build_model():
     global input_token_index,target_token_index
     encoder_inputs = Input(shape=(None, len(input_token_index)))
     encoder = LSTM(latent_dim, return_state=True)
     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
     encoder_states = [state_h, state_c]
     decoder_inputs = Input(shape=(None, len(target_token_index)))
     decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True)
     decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
     decoder_dense = Dense(len(target_token_index), activation='softmax')
     decoder_outputs = decoder_dense(decoder_outputs)
     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
     # 新序列预测时需要的encoder
     encoder_model = Model(encoder_inputs, encoder_states)
     # 新序列预测时需要的decoder
     decoder_state_input_h = Input(shape=(latent_dim,))
     decoder_state_input_c = Input(shape=(latent_dim,))
     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
     decoder_states = [state_h, state_c]
     decoder_outputs = decoder_dense(decoder_outputs)
     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

     return model, encoder_model, decoder_model

# 训练
if __name__ == "__main__":
     data_deal()
     model,encoder_model,decoder_model = build_model()
     model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
     model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
     model.save('model.h5')
     encoder_model.save('encoder_model.h5')
     decoder_model.save('decoder_model.h5')
