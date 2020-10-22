import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM
import numpy as np
print('TensorFlow: ', tf.__version__)


vocab_size = 10000
LSTM_DIM = 256
BATCH_SIZE = 16
embedding_matrix = np.random.randn(vocab_size, 300)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix, n_units, batch_size):
        super(Encoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        self.embedding = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True, mask_zero=True)
        self.lstm = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")


    @tf.function
    def call(self, inputs):
        input_utterence, initial_state = inputs
        input_embed = self.embedding(input_utterence)
        encoder_states, h1, c1 = self.lstm(input_embed, initial_state=initial_state)
        return encoder_states, h1, c1


    def create_initial_state(self):
        return tf.zeros((self.batch_size, self.n_units))





random_input = tf.random.uniform(shape=[BATCH_SIZE, 3], maxval=vocab_size, dtype=tf.int32)


encoder = Encoder(vocab_size, embedding_matrix, LSTM_DIM, BATCH_SIZE)




encoder_initial_state = [encoder.create_initial_state(), encoder.create_initial_state()]

_ = encoder([random_input, encoder_initial_state]) # required so that encoder.build is triggered


tf.saved_model.save(encoder, "encoder_model", signatures=encoder.call.get_concrete_function(
    [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_utterence'),
        [
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='initial_h'),
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='initial_c')
        ]
    ]))

loaded_model = tf.saved_model.load('encoder_model')
loaded_model([random_input, encoder_initial_state])