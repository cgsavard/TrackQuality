import numpy as np
import keras2onnx
import onnxruntime

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
X = np.array(np.random.rand(10, 21), dtype=np.float32)
print(model.predict(X))

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'NN_model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
sess = onnxruntime.InferenceSession(temp_model_file)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

print(sess.get_inputs()[0].name)
print(label_name)

# retrieve prediction - passing in the input list (you can also pass in multiple inputs as a list of lists)
pred_onx = sess.run([label_name], {input_name: X[0:1]})[0]
print(pred_onx)

