import tensorflow as tf
import numpy as np

class CustomModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x


model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save("my_model")

# Delete the custom-defined model class to ensure that the loader does not have
# access to it.
del CustomModel

loaded = tf.keras.models.load_model("my_model")
np.testing.assert_allclose(loaded(input_arr), outputs)

print("Original model:", model)
print("Loaded model:", loaded)
