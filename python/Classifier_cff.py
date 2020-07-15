import FWCore.ParameterSet.Config as cms

TrackClassifier = cms.EDProducer("L1TrackClassifier",
                                  L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"), 
                                  Algorithm = cms.string("None"),
                                  NNIdGraph = cms.string("L1Trigger/TrackFindingTracklet/data/FakeIDnet"),
                                  GBDTIdONNXmodel = cms.string("L1Trigger/TrackFindingTracklet/data/FakeIDGBDT/saved_model.onnx"),
                                  maxZ0 = cms.double ( 15. ) ,    # in cm
                                  maxEta = cms.double ( 2.4 ) ,
                                  chi2dofMax = cms.double( 40. ),
                                  bendchi2Max = cms.double( 2.4 ),
                                  minPt = cms.double( 2. ),       # in GeV
                                  nStubsmin = cms.int32( 4 ),
                                  nfeatures = cms.int32(21),                                    
    )