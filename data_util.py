from keras.models import Model
from keras.layers import Input,LSTM,Dense
import numpy as np
import pandas as pd

num_samples = 5000
# 定义路径
question_path = 'question.txt'
answer_path = 'answer.txt'


max_encoder_seq_length = None
max_decoder_seq_length = None
num_encoder_tokens = None
num_decoder_tokens = None

def get_xy_data():
    input_texts = []
    target_texts = []
    with open(question_path, 'r', encoding='utf-8') as f:
        input_texts = f.read().split('\n')
        input_texts = input_texts[:min(num_samples,len(input_texts)-1)]
    with open(answer_path, 'r', encoding='utf-8') as f:
        target_texts = ['\t' + line  + '\n' for line in f.read().split('\n')]
        target_texts = target_texts[:min(num_samples,len(input_texts)-1)]
    
    return input_texts, target_texts

def get_vocab_dict(X, Y):
    global max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens
    input_texts = X
    target_texts = Y
    input_characters = set()
    target_characters = set()
    for line in input_texts[:min(num_samples,len(input_texts)-1)]:
        for char in line:
            if char not in input_characters:
                input_characters.add(char)
    for line in target_texts[:min(num_samples,len(target_texts)-1)]:
        for char in line:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_encoder_seq_length)
    
    input_token_index = dict(
            [(char,i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
            [(char,i) for i, char in enumerate(target_characters)])
    
    return input_token_index, target_token_index

def get_rev_dict(input_token_index, target_token_index):
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index  = dict(
        (i, char) for char, i in target_token_index.items())
    return reverse_input_char_index, reverse_target_char_index