/*
 * L1TrackClassifier
 *
 * An ED producer taking L1 TTTracks without MVA fields filled
 * Returning L! TTTracks with MVA fields filled 
 * 
 * Uses pretrained ML models to classify tracks
 *
 *  Created on: July 15, 2020
 *      Author: Christopher Brown
 */

#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "L1Trigger/TrackQuality/interface/FeatureTransform.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


using namespace std;


class L1TrackClassifier : public edm::EDProducer {
public:

  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TrackClassifier(const edm::ParameterSet&);
  ~L1TrackClassifier();

private:
  virtual void beginJob() ;
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  string algorithm;

  float cut_min_pt_;
  float cut_max_z0_ ;
  float cut_max_eta_ ;
  float cut_max_chi2_ ;
  float cut_max_bendchi_;
  int cut_min_nstubs_;

  float trk_pt;
  float trk_bend_chi2;
  float trk_z0;
  float trk_eta;
  float trk_chi2;
  int nStubs;
  

  vector<float> TransformedFeatures;
  vector<string> in_features;
  int n_features;

  string ONNX_path;
  string TF_path;

  
  // FloatArray type defined in https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h
  // as: std::vector<std::vector<float>> FloatArrays;
  cms::Ort::FloatArrays ortinput;
  vector<string> ortinput_names;
  cms::Ort::FloatArrays ortoutputs;
  vector<string> ortoutput_names;

  


  
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;

};

///////////////
//constructor//
///////////////
L1TrackClassifier::L1TrackClassifier(const edm::ParameterSet& iConfig) :
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))){
  

  algorithm = (string)iConfig.getParameter<string>("Algorithm");

  if ((algorithm == "Cut") | (algorithm == "All") ) {
    // Track MET purity cut is included for comparision
    cut_min_pt_ = (float)iConfig.getParameter<double>("minPt");
    cut_max_z0_ = (float)iConfig.getParameter<double>("maxZ0");
    cut_max_eta_ = (float)iConfig.getParameter<double>("maxEta");
    cut_max_chi2_ = (float)iConfig.getParameter<double>("chi2dofMax");
    cut_max_bendchi_ = (float)iConfig.getParameter<double>("bendchi2Max");
    cut_min_nstubs_ = (int)iConfig.getParameter<int>("nStubsmin");
            
  }

  
  if ((algorithm == "GBDT") | (algorithm == "NN") | (algorithm == "All")) {

    in_features = iConfig.getParameter<vector<string>>("in_features");

    n_features = in_features.size();
    // ONNX Neural Net and GBDT implementation

    if ((algorithm == "GBDT") | (algorithm == "All")){
      ONNX_path = edm::FileInPath(iConfig.getParameter<string>("GBDTIdONNXmodel")).fullPath();
      ortinput_names.push_back(iConfig.getParameter<string>("GBDTIdONNXInputName"));
    }
    if (algorithm == "NN") {
      ONNX_path = edm::FileInPath(iConfig.getParameter<string>("NNIdONNXmodel")).fullPath();
      ortinput_names.push_back(iConfig.getParameter<string>("NNIdONNXInputName"));
    }
    cout << "loading fake ID onnx model from " << ONNX_path << std::endl;
  
  }

  produces< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >( "Level1TTTracks" ).setBranchAlias("Level1TTTracks");
}

//////////////
//destructor//
//////////////
L1TrackClassifier::~L1TrackClassifier()
{

}

////////////
//producer//
////////////
void L1TrackClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  //Get TTTracks
  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;
  
  // Prepare output TTTracks
  std::unique_ptr< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > L1TkTracksForOutput( new std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > );
  cout << algorithm << endl;
  //Iterate through tracks
  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {

    TTTrack< Ref_Phase2TrackerDigi_ > aTrack = *trackIter;
        
    if ((algorithm == "Cut") | (algorithm == "All")) {
      trk_pt = aTrack.momentum().perp();
      trk_bend_chi2 = aTrack.stubPtConsistency();
      trk_z0 = aTrack.z0();
      trk_eta = aTrack.momentum().eta();
      trk_chi2 = aTrack.chi2();
      const auto& stubRefs = aTrack.getStubRefs();
      nStubs = stubRefs.size();

      float classification = 0.0; // Default classification is 0

      if (trk_pt >= cut_min_pt_ && 
        abs(trk_z0) < cut_max_z0_ && 
        abs(trk_eta) < cut_max_eta_ && 
        trk_chi2 < cut_max_chi2_ && 
        trk_bend_chi2 < cut_max_bendchi_ && 
        nStubs >= cut_min_nstubs_) classification = 1.0;
        // Classification updated to 1 if conditions are met

        aTrack.settrkMVA1(classification);
    }


    if ((algorithm == "GBDT") | (algorithm == "NN") | (algorithm == "All")) {
      
      TransformedFeatures = FeatureTransform::Transform(aTrack,in_features); //Transform feautres
      cms::Ort::ONNXRuntime Runtime(ONNX_path); //Setup ONNX runtime

      //ONNX runtime recieves a vector of vectors of floats so push back the input
      // vector of float to create a 1,1,21 ortinput
      ortinput.push_back(TransformedFeatures);

      // batch_size 1 as only one set of transformed features is being processed
      int batch_size = 1;
      // Run classification on a batch of 1
      ortoutput_names = Runtime.getOutputNames();
      ortoutputs = Runtime.run(ortinput_names,ortinput,ortoutput_names,batch_size); 
      // access first value of nested vector
      if (algorithm == "NN"){
        aTrack.settrkMVA1(ortoutputs[0][0]);
      }

      // The ortoutput_names vector for the GBDT is left blank due to issues returning the correct
      // output, instead the GBDT will fill the ortoutputs with both the class prediciton and the class 
      // probabilities. 
      //ortoutputs[0][0] = class prediction based on a 0.5 threshold
      //ortoutputs[1][0] = negative class probability
      //ortoutputs[1][1] = positive class probability
      
      if (algorithm == "GBDT"){
        aTrack.settrkMVA1(ortoutputs[1][1]);
      }

      if (algorithm == "All"){
        aTrack.settrkMVA3(ortoutputs[1][1]);
      }
      
      // remove previous transformed feature ready for next track
      ortinput.pop_back();
    
    }
  
    else if ((algorithm == "None")){
      // Default no algorithm
      aTrack.settrkMVA1(-999);
      aTrack.settrkMVA2(-999);
      aTrack.settrkMVA3(-999);
    }
        
    aTrack.setTrackWordBits();
    L1TkTracksForOutput->push_back(aTrack);

  }
  
  iEvent.put( move(L1TkTracksForOutput), "Level1TTTracks");

}

  

// end producer

void L1TrackClassifier::beginJob() {
}

void L1TrackClassifier::endJob() {

}

DEFINE_FWK_MODULE(L1TrackClassifier);
