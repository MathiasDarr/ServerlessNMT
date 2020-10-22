import tensorflow as tf


class Model(tf.keras.Model):

    @tf.function
    def call(self, x):
        ...

m = Model()
tf.saved_model.save(m, '/tmp/saved_model/',
        signatures=m.call.get_concrete_function(tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="inp")))

