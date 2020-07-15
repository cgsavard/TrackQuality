import numpy as np
import xgboost as xgb
import joblib
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

X = np.array(np.random.rand(10, 21), dtype=np.float32)
model = joblib.load("Classifier.pkl")
print(model.predict(X))

num_features = 21
initial_type = [('feature_input', FloatTensorType([1, num_features]))]
 
onx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)

# Save your model locally (or where you desire!)
with open("test.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt

# setup runtime - load the persisted ONNX model
sess = rt.InferenceSession("test.onnx")

# get model metadata to enable mapping of new input to the runtime model.
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

print(sess.get_inputs()[0].name)
print(label_name)

# retrieve prediction - passing in the input list (you can also pass in multiple inputs as a list of lists)
pred_onx = sess.run([label_name], {input_name: X[0:1]})[0]
print(pred_onx)
