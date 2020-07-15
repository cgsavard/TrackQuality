'''
Helper script that converts a tensorflow 2.0 model saved in the h5 format to the metagraph format
It will produce a folder of name "FakeIDNNGraph" that contains the metagraph
'''

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np


model = load_model('Final_model.h5')
num_features = 21

random_input = np.random.rand(10,num_features)
print(model.predict(random_input))  #Used to check pre saving and post saving models

model.save("FakeIDNNGraph")

new_model = tf.keras.models.load_model("FakeIDNNGraph")
print(new_model.predict(random_input))

# Check its architecture
new_model.summary()

'''
 Names of input and output layers are hardcoded, these are needed in Classifier_cff as NNIdGraphInputName and NNIdGraphOutputName
  to extract them run:
        saved_model_cli show --dir [METAGRAPH SAVE DIR] --tag_set serve --signature_def serving_default 
        With example output: 
        The given SavedModel SignatureDef contains the following input(s):
          inputs['input_1'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 21)
            name: serving_default_input_1:0
        The given SavedModel SignatureDef contains the following output(s):
          outputs['Sigmoid_Output_Layer'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 1)
            name: StatefulPartitionedCall:0
        Method name is: tensorflow/serving/predict
      
'''