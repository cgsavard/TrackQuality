import FWCore.ParameterSet.Config as cms

TrackClassifier = cms.EDProducer("L1TrackClassifier",
                                  L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"), 
                                  Algorithm = cms.string("None"), #None, Cut, NN, GBDT

                                  NNIdONNXmodel = cms.string("L1Trigger/TrackQuality/data/FakeIDNN/NN_model.onnx"),
                                  NNIdONNXInputName = cms.string("input_1"),

                                  GBDTIdONNXmodel = cms.string("L1Trigger/TrackQuality/data/FakeIDGBDT/GBDT_model.onnx"),
                                  GBDTIdONNXInputName = cms.string("feature_input"),

                                  in_features = cms.vstring(["log_chi2","log_bendchi2","log_chi2rphi","log_chi2rz",
                                                             "nstubs","lay1_hits","lay2_hits","lay3_hits","lay4_hits",
                                                             "lay5_hits","lay6_hits","disk1_hits","disk2_hits",
                                                             "disk3_hits","disk4_hits","disk5_hits","rinv","tanl",
                                                             "z0","dtot","ltot"]),

                                  maxZ0 = cms.double ( 15. ) ,    # in cm
                                  maxEta = cms.double ( 2.4 ) ,
                                  chi2dofMax = cms.double( 40. ),
                                  bendchi2Max = cms.double( 2.4 ),
                                  minPt = cms.double( 2. ),       # in GeV
                                  nStubsmin = cms.int32( 4 ),
                                  
    )
