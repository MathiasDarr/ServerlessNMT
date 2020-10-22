import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM
import numpy as np

vocab_size = 10000
LSTM_DIM = 256
BATCH_SIZE = 16
embedding_matrix = np.random.randn(vocab_size, 300)

# class Encoder(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, n_units, batch_size):
#         super(Encoder, self).__init__()
#         self.n_units = n_units
#         self.batch_size = batch_size
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#         # self.embedding = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True, mask_zero=True)
#         self.lstm = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
#
#
#     @tf.function
#     def call(self, inputs):
#         input_utterence, initial_state = inputs
#         input_embed = self.embedding(input_utterence)
#         encoder_states, h1, c1 = self.lstm(input_embed, initial_state=initial_state)
#         return encoder_states, h1, c1
#
#     def create_initial_state(self):
#         return tf.zeros((self.batch_size, self.n_units))



class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    @tf.function
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    def create_initial_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))