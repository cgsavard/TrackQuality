# TrackQuality
ED producer for adding track quality to the MVA field using ML pretrained models

## Structure

### util
Contains 3 conversion scripts:
* kerastoonnx.py which has prerequisites of the keras2onnx package and onnxruntime packages both part of the onnx libaray: https://github.com/onnx/onnx, and converts a pretrained keras model to the onnx format
* xgboosttoonnx.py which converts a pretrained XGBoost model to the onnx format
* kerastotfgraph.py which converts a pretrained keras model to the metagraph format, there could be compatability issues with TF1 vs TF2, this script has only been used for models trained in TF2

Contains the TTTrack.h file that has 3 new functions used to set the 3 MVA fields of the TTTrack, found in DataFormats/L1TrackTrigger/interface


### data
Contains pretrained models saved in the metagraph format for tensorflow and ONNX formats, the contents of this folder can be produced with scripts detailed in the util folder

### interface
Header file for feature transform function

### src
Source file for feature transform function used to tranform TTTrack variables to input features for ML models, specific to the model being tested

### plugins
Contains the ED producer that takes TTTracks and returns TTTracks with their MVA field filled

### python 
Contains the Classifier_cff file used to specify the parameters of the ED producer

### test
contains the L1TrackClassNtupleMaker ED analyser and config file used to generate NTuples with 3 new fields, MVA1,2,3 filled. Currently the ED producer only fills MVA1 but the functionality is there to fill all three and compare


## Running


This TrackQuality folder should be placed in the L1Trigger directory with the TTTrack.h in the DataFormats directory before being built and then run using the cmsRun L1TrackClassNtupleMaker_cfg.py
