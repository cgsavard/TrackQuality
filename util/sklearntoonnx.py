'''
Helper script that converts a XGBoost model saved in the pkl format to the onnx format
It will produce a file of name GBDT_model.onnx
'''

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# load in sklearn model
num_features = 6
X = np.array(np.random.rand(10, num_features), dtype=np.float32)
model = joblib.load("sklearn_GBDT.pkl")
print(model.predict_proba(X))

# convert sklearn to onnx and save
initial_type = [('float_input', FloatTensorType([None, num_features]))]
onx = convert_sklearn(model, initial_types=initial_type)
fname = "skl_GBDT.onnx"
with open(fname, "wb") as f:
    f.write(onx.SerializeToString())

# test onnx saved model
sess = rt.InferenceSession(fname)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[1].name #index 0 for prediction, 1 for probability
#pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

print(sess.get_inputs()[0].name)
print(label_name)

# predict on random input and compare to previous XGBoost model
for i in range(len(X)):
    pred_onx = sess.run([], {input_name: X[i:i+1]})[1]
    print(pred_onx)

