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

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "L1Trigger/TrackFindingTracklet/interface/FeatureTransform.h"
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
  

  vector<float> TransformedFeatures;
  int n_features;

  string ONNX_path;
  string TF_path;

  // Default pointers to tensorflow MetaGraph and session
  tensorflow::MetaGraphDef* FakeIDGraph_;
  tensorflow::Session* FakeIDSesh_;
  // Output of a tensorflow tensor
  vector<tensorflow::Tensor> tfoutput;


  Ort::SessionOptions* session_options; //Default session options
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

  if (algorithm == "Cut") {
    // Track MET purity cut is included for comparision
    float cut_min_pt_ = (float)iConfig.getParameter<double>("minPt");
    float cut_max_z0_ = (float)iConfig.getParameter<double>("maxZ0");
    float cut_max_eta_ = (float)iConfig.getParameter<double>("maxEta");
    float cut_max_chi2_ = (float)iConfig.getParameter<double>("chi2dofMax");
    float cut_max_bendchi_ = (float)iConfig.getParameter<double>("bendchi2Max");
    int cut_min_nstubs_ = (int)iConfig.getParameter<int>("nStubsmin");
            
  }

  if (algorithm == "TFNN") {
    // TensorFlow Neural Net implementation
    n_features = iConfig.getParameter<int>("nfeatures");
    TF_path = iConfig.getParameter<string>("NNIdGraph");

    string tf_input_name = iConfig.getParameter<string>("NNIdGraphInputName");
    string tf_output_name = iConfig.getParameter<string>("NNIdGraphOutputName");

    cout << "loading fake ID NN tensorflow graph from " << TF_path << std::endl;
    // load the graph
    FakeIDGraph_ = tensorflow::loadMetaGraphDef(TF_path,"serve");
    // create a new session and add the graphDef
    FakeIDSesh_ = tensorflow::createSession(FakeIDGraph_,TF_path);

  }

  if ((algorithm == "GBDT") | (algorithm == "OXNN")) {
    // ONNX Neural Net and GBDT implementation
    n_features = iConfig.getParameter<int>("nfeatures");
    if (algorithm == "GBDT") {
      ONNX_path = edm::FileInPath(iConfig.getParameter<string>("GBDTIdONNXmodel")).fullPath();
      ortinput_names.push_back(iConfig.getParameter<string>("GBDTIdONNXInputName"));
      ortoutput_names.push_back(iConfig.getParameter<string>("GBDTIdONNXOutputName"));

    }
    if (algorithm == "OXNN") {
      ONNX_path = edm::FileInPath(iConfig.getParameter<string>("NNIdONNXmodel")).fullPath();
      ortinput_names.push_back(iConfig.getParameter<string>("NNIdONNXInputName"));
      ortoutput_names.push_back(iConfig.getParameter<string>("NNIdONNXOutputName"));
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

  //Iterate through tracks
  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {

    TTTrack< Ref_Phase2TrackerDigi_ > aTrack = *trackIter;
        
    if (algorithm == "Cut") {
      float trk_pt = aTrack.momentum().perp();
      float trk_bend_chi2 = aTrack.stubPtConsistency();
      float trk_z0 = aTrack.z0();
      float trk_eta = aTrack.momentum().eta();
      float trk_chi2 = aTrack.chi2();
      const auto& stubRefs = aTrack.getStubRefs();
      int nStubs = stubRefs.size();

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


    if (algorithm == "TFNN") {
      TransformedFeatures = FeatureTransform::Transform(aTrack); //Transform features
      tensorflow::Tensor tfinput(tensorflow::DT_FLOAT, { 1, n_features }); //Prepare input tensor
      
      for (int i=0;i<n_features;++i){
        // fill input tensor with transformed features
        tfinput.tensor<float, 2>()(0, i) = TransformedFeatures[i];
      }    
     
      //Run session filling tfouput tensor
      tensorflow::run(FakeIDSesh_ , { { tf_input_name, tfinput } }, { tf_output_name }, &tfoutput);

      // set track classification by accesing the output float of the tfouput tensor
      aTrack.settrkMVA1(tfoutput[0].tensor<float, 2>()(0, 0));
      

    }

    if ((algorithm == "GBDT") | (algorithm == "OXNN")) {
      TransformedFeatures = FeatureTransform::Transform(aTrack); //Transform feautres
      cms::Ort::ONNXRuntime Runtime(ONNX_path ,session_options); //Setup ONNX runtime

      //ONNX runtime recieves a vector of vectors of floats so push back the input
      // vector of float to create a 1,1,21 ortinput
      ortinput.push_back(TransformedFeatures);

      // batch_size 1 as only one set of transformed features is being processed
      int batch_size = 1;
      // Run classification on a batch of 1
      ortoutputs = Runtime.run(ortinput_names,ortinput,ortoutput_names,batch_size); 
      // access first value of nested vector
      aTrack.settrkMVA1(ortoutputs[1][0]);
      
      // remove previous transformed feature ready for next track
      ortinput.pop_back();
    
    }
  
    if (algorithm == "None"){
      // Default no algorithm
      aTrack.settrkMVA1(-999);
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
  if (algorithm == "TFNN") {
    //deleta the session
    tensorflow::closeSession(FakeIDSesh_);   
    FakeIDSesh_ = nullptr;


    // delete the graph	    
    delete FakeIDGraph_;	 
    FakeIDGraph_ = nullptr; 

  }

  if ((algorithm == "GBDT") | (algorithm == "OXNN")){
    delete session_options;
    session_options = nullptr;
  }
}

DEFINE_FWK_MODULE(L1TrackClassifier);