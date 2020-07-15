import FWCore.ParameterSet.Config as cms

TrackClassifier = cms.EDProducer("L1TrackClassifier",
                                  L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"), 
                                  Algorithm = cms.string("None"), #None, Cut, TFNN, OXNN, GBDT
                                  NNIdGraph = cms.string("L1Trigger/TrackQuality/data/FakeIDNNGraph"),
                                  NNIdGraphInputName = cms.string("serving_default_input_1"),
                                  NNIdGraphOutputName = cms.string("StatefulPartitionedCall"),
                                  NNIdONNXmodel = cms.string("L1Trigger/TrackQuality/data/FakeIDNN/NN_model.onnx"),
                                  NNIdONNXInputName = cms.string("input_1"),
                                  GBDTIdONNXmodel = cms.string("L1Trigger/TrackQuality/data/FakeIDGBDT/GBDT_model.onnx"),
                                  GBDTIdONNXInputName = cms.string("feature_input"),
                                  maxZ0 = cms.double ( 15. ) ,    # in cm
                                  maxEta = cms.double ( 2.4 ) ,
                                  chi2dofMax = cms.double( 40. ),
                                  bendchi2Max = cms.double( 2.4 ),
                                  minPt = cms.double( 2. ),       # in GeV
                                  nStubsmin = cms.int32( 4 ),
                                  nfeatures = cms.int32(21),                                    
    )