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
  float cut_min_pt_;
  float cut_max_z0_;
  float cut_max_eta_;
  float cut_max_chi2_;
  float cut_max_bendchi_;
  float cut_min_nstubs_;
 
  float trk_pt;
  float trk_bend_chi2;
  float trk_z0;
  float trk_eta;
  float trk_chi2;
  int nStubs;

  vector<float> TransformedFeatures;
  int n_features;

  string GBDT_path;
  string NN_path;

  tensorflow::MetaGraphDef* FakeIDGraph_;
  tensorflow::Session* FakeIDSesh_;
  vector<tensorflow::Tensor> tfoutput;

  Ort::SessionOptions* session_options;
  cms::Ort::FloatArrays ortinput;
  vector<string> ortinput_names;
  cms::Ort::FloatArrays ortoutputs;
  vector<string> ortoutput_names;

  float classification;


  
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;

};

///////////////
//constructor//
///////////////
L1TrackClassifier::L1TrackClassifier(const edm::ParameterSet& iConfig) :
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))){

  algorithm = (string)iConfig.getParameter<string>("Algorithm");

  if (algorithm == "Cut") {
    cut_min_pt_ = (float)iConfig.getParameter<double>("minPt");
    cut_max_z0_ = (float)iConfig.getParameter<double>("maxZ0");
    cut_max_eta_ = (float)iConfig.getParameter<double>("maxEta");
    cut_max_chi2_ = (float)iConfig.getParameter<double>("chi2dofMax");
    cut_max_bendchi_ = (float)iConfig.getParameter<double>("bendchi2Max");
    cut_min_nstubs_ = (int)iConfig.getParameter<int>("nStubsmin");

    cout << "Track MET purity cut will be Performed" << endl;
            
  }

  if (algorithm == "NN") {
    cout << "Neural Network Classification will be Performed" << endl;
    n_features = iConfig.getParameter<int>("nfeatures");
    NN_path = iConfig.getParameter<string>("NNIdGraph");
    cout << "loading fake ID NN tensorflow graph from " << NN_path << std::endl;
    // load the graph
    FakeIDGraph_ = tensorflow::loadMetaGraphDef(NN_path,"serve");
    // create a new session and add the graphDef
    FakeIDSesh_ = tensorflow::createSession(FakeIDGraph_,NN_path);

  }

  if (algorithm == "GBDT"){
    cout << "GBDT Classification will be Performed" << endl;
    n_features = iConfig.getParameter<int>("nfeatures");
    GBDT_path = edm::FileInPath(iConfig.getParameter<string>("GBDTIdONNXmodel")).fullPath();
    cout << "loading fake ID GBDT onnx model from " << GBDT_path << std::endl;
    ortinput_names.push_back("feature_input");
  
  }

  else if (algorithm == "None"){
    cout << "No Classification will be Performed" << endl;
  }


  produces< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >( "Level1ClassTTTracks" ).setBranchAlias("Level1ClassTTTracks");
}

L1TrackClassifier::~L1TrackClassifier()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

////////////
//producer//
////////////
void L1TrackClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  std::unique_ptr< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > L1TkTracksForOutput( new std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > );

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {

    TTTrack< Ref_Phase2TrackerDigi_ > aTrack = *trackIter;
        
    if (algorithm == "Cut") {
      trk_pt = aTrack.momentum().perp();
      trk_bend_chi2 = aTrack.stubPtConsistency();
      trk_z0 = aTrack.z0();
      trk_eta = aTrack.momentum().eta();
      trk_chi2 = aTrack.chi2();
      const auto& stubRefs = aTrack.getStubRefs();
      nStubs = stubRefs.size();

      classification = 0.0;

      if (trk_pt >= cut_min_pt_ && 
        abs(trk_z0) < cut_max_z0_ && 
        abs(trk_eta) < cut_max_eta_ && 
        trk_chi2 < cut_max_chi2_ && 
        trk_bend_chi2 < cut_max_bendchi_ && 
        nStubs >= cut_min_nstubs_) classification = 1.0;

        aTrack.settrkMVA1(classification);
    }


    if (algorithm == "NN") {
      TransformedFeatures = FeatureTransform::Transform(aTrack);
      tensorflow::Tensor tfinput(tensorflow::DT_FLOAT, { 1, n_features });
      
      for (int i=0;i<n_features;++i){
        tfinput.tensor<float, 2>()(0, i) = TransformedFeatures[i];
      }    
      tensorflow::run(FakeIDSesh_ , { { "serving_default_input_1", tfinput } }, { "StatefulPartitionedCall" }, &tfoutput);
      // set track classification


      aTrack.settrkMVA1(tfoutput[0].tensor<float, 2>()(0, 0));
      

    }

    if (algorithm == "GBDT") {
      TransformedFeatures = FeatureTransform::Transform(aTrack);
      cms::Ort::ONNXRuntime Runtime(GBDT_path ,session_options);

      ortinput.push_back(TransformedFeatures);

      int batch_size = 1;
      ortoutputs = Runtime.run(ortinput_names,ortinput,ortoutput_names,batch_size);
      aTrack.settrkMVA1(ortoutputs[1][0]);
      

      ortinput.pop_back();
    
    }
  
    if (algorithm == "None"){
      aTrack.settrkMVA1(-999);
    }
        
    aTrack.setTrackWordBits();
    L1TkTracksForOutput->push_back(aTrack);

  }
  
  iEvent.put( move(L1TkTracksForOutput), "Level1ClassTTTracks");

}

  

// end producer

void L1TrackClassifier::beginJob() {
}

void L1TrackClassifier::endJob() {
  if ((algorithm == "NN") | (algorithm == "All")){
    //deleta the session
    tensorflow::closeSession(FakeIDSesh_);   
    FakeIDSesh_ = nullptr;


    // delete the graph	    
    delete FakeIDGraph_;	 
    FakeIDGraph_ = nullptr; 

  }

  if ((algorithm == "GBDT") | (algorithm == "All")){
    delete session_options;
    session_options = nullptr;
  }
}

DEFINE_FWK_MODULE(L1TrackClassifier);