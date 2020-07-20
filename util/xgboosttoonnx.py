'''
Helper script that converts a XGBoost model saved in the pkl format to the onnx format
It will produce a file of name GBDT_model.onnx
'''

import numpy as np
import xgboost as xgb
import joblib
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

num_features = 21
X = np.array(np.random.rand(10, num_features), dtype=np.float32)
model = joblib.load("GBDTClassv1.pkl")
#print(model.predict(X))

# The name of the input is needed in Clasifier_cff as GBDTIdONNXInputName
initial_type = [('feature_input', FloatTensorType([1, num_features]))]

 
onx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)

# Save the model
with open("GBDT_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# This tests the model
import onnxruntime as rt

# setup runtime - load the persisted ONNX model
sess = rt.InferenceSession("GBDT_model.onnx")

# get model metadata to enable mapping of new input to the runtime model.
input_name = sess.get_inputs()[0].name
# This label will access the class probabilities when run in CMSSW, use index 0 for class prediction
label_name = sess.get_outputs()[1].name


print(sess.get_inputs()[0].name)
# The name of the output is needed in Clasifier_cff as GBDTIdONNXOutputName
print(label_name)

# predict on random input and compare to previous XGBoost model
for i in range(len(X)):
    pred_onx = sess.run([], {input_name: X[i:i+1]})[1]
    print(pred_onx)
