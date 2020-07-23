/*
Function to transform TTTrackWord variables into those used by ML models
Inputs a TTTrack and returns a vector of floats of dimension (1,n_training_features)
This file is specific to the training of the ML model and should be adapted accordingly
*/
#include "L1Trigger/TrackQuality/interface/FeatureTransform.h"
#include <iostream>
#include <memory>
#include <map>
#include <string>

namespace FeatureTransform {


std::vector<float> Transform(TTTrack < Ref_Phase2TrackerDigi_ > aTrack, std::vector<std::string> in_features){

    // List input features for MVA in proper order below, the features options are 
    // {"log_chi2","log_chi2rphi","log_chi2rz","log_bendchi2","nstubs","lay1_hits","lay2_hits",
    // "lay3_hits","lay4_hits","lay5_hits","lay6_hits","disk1_hits","disk2_hits","disk3_hits",
    // "disk4_hits","disk5_hits","rinv","tanl","z0","dtot","ltot","chi2","chi2rz","chi2rphi",
    // "bendchi2","pt","eta","nlaymiss_interior"}
    
    std::vector<float> transformed_features;

    // The following converts the 7 bit hitmask in the TTTrackword to an expected
    // 11 bit hitmask based on the eta of the track
    std::vector<int> hitpattern_binary = {0,0,0,0,0,0,0};
    std::vector<int> hitpattern_expanded_binary = {0,0,0,0,0,0,0,0,0,0,0,0};
    std::vector<float> eta_bins = {0.0,0.2,0.41,0.62,0.9,1.26,1.68,2.08,2.4};


    // Expected hitmap table, each row corresponds to an eta bin, each value corresponds to 
    // the expected layer in the expanded hit pattern. The expanded hit pattern should be
    // 11 bits but contains a 12th element so this hitmap table is symmetric, the 12th element 
    // is removed.
    int hitmap[8][7] = {{0, 1,  2,  3,  4,  5,  11},
                        {0, 1,  2,  3,  4,  5,  11},
                        {0, 1,  2,  3,  4,  5,  11},
                        {0, 1,  2,  3,  4,  5,  11},
                        {0, 1,  2,  3,  4,  5,  11},
                        {0, 1,  2,  6,  7,  8,  9 },
                        {0, 1,  7,  8,  9, 10,  11},
                        {0, 6,  7,  8,  9, 10,  11}};

    // iterate through bits of the hitpattern and compare to 1 filling the hitpattern binary vector
    int tmp_trk_hitpattern = aTrack.hitPattern();
    for (int i = 6; i >=0; i--){
      int k = tmp_trk_hitpattern >> i;
            if (k&1)
                hitpattern_binary[i] = 1;
        }

    // calculate number of missed interior layers from hitpattern
    int nbits = floor(log2(tmp_trk_hitpattern)) + 1;
    int lay_i = 0;
    int tmp_trk_nlaymiss_interior = 0;
    bool seq = 0;
    for (int i=0; i<nbits; i++){
      lay_i = ((1<<i)&tmp_trk_hitpattern)>>i; //0 or 1 in ith bit (right to left)
    
      if (lay_i && !seq) seq = 1; //sequence starts when first 1 found
      if (!lay_i && seq) tmp_trk_nlaymiss_interior++;
    }


    float eta = abs(aTrack.eta());
    int eta_size = static_cast<int>(eta_bins.size());
    // First iterate through eta bins

    for (int j=0; j<eta_size; j++)
        {
          if (eta >= eta_bins[j] && eta < eta_bins[j+1]) // if track in eta bin
          {
              // Iterate through hitpattern binary
              for (int k=0; k<6; k++)
                  // Fill expanded binary entries using the expected hitmap table positions 
                  hitpattern_expanded_binary[hitmap[j][k]] = hitpattern_binary[k];   
          }
        }

    hitpattern_expanded_binary.pop_back(); //remove final unused bit
    int tmp_trk_ltot;
    //calculate number of layer hits
    for (int i=0; i<6; ++i)
    {
      tmp_trk_ltot += hitpattern_expanded_binary[i];
    }
    

    int tmp_trk_dtot;
    //calculate number of disk hits
    for (int i=6; i<11; ++i)
    {
      tmp_trk_dtot += hitpattern_expanded_binary[i];
    }

    
    // While not strictly necessary to define these parameters,
    // it is included so each variable is named to avoid confusion
    float tmp_trk_big_invr   = 500*abs(aTrack.rInv());
    float tmp_trk_tanl  = abs(aTrack.tanL());
    float tmp_trk_z0   = abs( aTrack.z0() );
    float tmp_trk_pt = aTrack.momentum().perp();
    float tmp_trk_eta = aTrack.eta();
    float tmp_trk_chi2 = aTrack.chi2();
    float tmp_trk_chi2rphi = aTrack.chi2XY();
    float tmp_trk_chi2rz = aTrack.chi2Z();
    float tmp_trk_bendchi2 = aTrack.stubPtConsistency();
    float tmp_trk_log_chi2 = log(tmp_trk_chi2);
    float tmp_trk_log_chi2rphi = log(tmp_trk_chi2rphi);
    float tmp_trk_log_chi2rz = log(tmp_trk_chi2rz);
    float tmp_trk_log_bendchi2 = log(tmp_trk_bendchi2);

    // fill feature map
    std::map<std::string,float> feat_map; 
    feat_map["log_chi2"] = tmp_trk_log_chi2;
    feat_map["log_chi2rphi"] = tmp_trk_log_chi2rphi;
    feat_map["log_chi2rz"] = tmp_trk_log_chi2rz;
    feat_map["log_bendchi2"] = tmp_trk_log_bendchi2;
    feat_map["chi2"] = tmp_trk_chi2;
    feat_map["chi2rphi"] = tmp_trk_chi2rphi;
    feat_map["chi2rz"] = tmp_trk_chi2rz;
    feat_map["bendchi2"] = tmp_trk_bendchi2;
    feat_map["nstubs"] = float(tmp_trk_dtot+tmp_trk_ltot);
    feat_map["lay1_hits"] = float(hitpattern_expanded_binary[0]);
    feat_map["lay2_hits"] = float(hitpattern_expanded_binary[1]);
    feat_map["lay3_hits"] = float(hitpattern_expanded_binary[2]);
    feat_map["lay4_hits"] = float(hitpattern_expanded_binary[3]);
    feat_map["lay5_hits"] = float(hitpattern_expanded_binary[4]);
    feat_map["lay6_hits"] = float(hitpattern_expanded_binary[5]);
    feat_map["disk1_hits"] = float(hitpattern_expanded_binary[6]);
    feat_map["disk2_hits"] = float(hitpattern_expanded_binary[7]);
    feat_map["disk3_hits"] = float(hitpattern_expanded_binary[8]);
    feat_map["disk4_hits"] = float(hitpattern_expanded_binary[9]);
    feat_map["disk5_hits"] = float(hitpattern_expanded_binary[10]);
    feat_map["rinv"] = tmp_trk_big_invr;
    feat_map["tanl"] = tmp_trk_tanl;
    feat_map["z0"] = tmp_trk_z0;
    feat_map["dtot"] = float(tmp_trk_dtot);
    feat_map["ltot"] = float(tmp_trk_ltot);
    feat_map["pt"] = tmp_trk_pt;
    feat_map["eta"] = tmp_trk_eta;
    feat_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);

    // fill tensor with track params
    for (std::string feat : in_features) 
      transformed_features.push_back(feat_map[feat]);
    /*transformed_features.push_back(float(tmp_trk_log_chi2));
    transformed_features.push_back(float(tmp_trk_log_chi2rphi));
    transformed_features.push_back(float(tmp_trk_log_chi2rz));
    transformed_features.push_back(float(tmp_trk_log_bendchi2));
    transformed_features.push_back(float(tmp_trk_dtot+tmp_trk_ltot)); //nstubs
    transformed_features.push_back(float(hitpattern_expanded_binary[0]));
    transformed_features.push_back(float(hitpattern_expanded_binary[1]));
    transformed_features.push_back(float(hitpattern_expanded_binary[2]));
    transformed_features.push_back(float(hitpattern_expanded_binary[3]));
    transformed_features.push_back(float(hitpattern_expanded_binary[4]));
    transformed_features.push_back(float(hitpattern_expanded_binary[5]));
    transformed_features.push_back(float(hitpattern_expanded_binary[6]));
    transformed_features.push_back(float(hitpattern_expanded_binary[7]));
    transformed_features.push_back(float(hitpattern_expanded_binary[8]));
    transformed_features.push_back(float(hitpattern_expanded_binary[9]));
    transformed_features.push_back(float(hitpattern_expanded_binary[10]));
    transformed_features.push_back(float(tmp_trk_big_invr));
    transformed_features.push_back(float(tmp_trk_tanl));
    transformed_features.push_back(float(tmp_trk_z0));
    transformed_features.push_back(float(tmp_trk_dtot));
    transformed_features.push_back(float(tmp_trk_ltot));*/

    return transformed_features;

    }
}

