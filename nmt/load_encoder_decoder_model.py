import tensorflow as tf
from nmt.models.decoder import Decoder
from nmt.models.encoder import Encoder


def load_encoder_decoder():
    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    vocab_inp_size = 7591
    vocab_tar_size = 4367

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    # encoder.load_weights('encoder_model_weights.h5')
    # decoder.load_weights('decoder_model_weights.h5')
    return encoder, decoder

encoder, decoder = load_encoder_decoder()
data