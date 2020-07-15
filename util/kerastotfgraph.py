import tensorflow as tf
from keras import backend as K

from tensorflow.keras.models import load_model
from constraints import ZeroSomeWeights
from keras.utils.generic_utils import get_custom_objects
import numpy as np
get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})

model = load_model('Final_model.h5',custom_objects={'ZeroSomeWeights':ZeroSomeWeights})
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]
random_input = np.random.rand(1,21)
print(model.predict(random_input))

model.save("FakeIDnet")

new_model = tf.keras.models.load_model("FakeIDnet")
print(new_model.predict(random_input))

# Check its architecture
new_model.summary()
print(new_model.outputs)
print(new_model.inputs)