/*
Function to transform TTTrackWord variables into those used by ML models
Inputs a TTTrack and returns a vector of floats of dimension (1,n_training_features)
This file is specific to the training of the ML model and should be adapted accordingly
*/
#include "L1Trigger/TrackFindingTracklet/interface/FeatureTransform.h"
#include <iostream>
#include <memory>


namespace FeatureTransform {


std::vector<float> Transform(TTTrack < Ref_Phase2TrackerDigi_ > aTrack){

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
    for (int i = 6; i >=0; i--){
        int k = aTrack.hitPattern() >> i;
            if (k&1)
                hitpattern_binary[i] = 1;
        }


    float eta = abs(aTrack.eta());
    // First iterate through eta bins
    for (int j=0; j<eta_bins.size(); j++)
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
    float tmp_trk_log_chi2 = log(aTrack.chi2());
    float tmp_trk_log_chi2rphi = log(aTrack.chi2XY());
    float tmp_trk_log_chi2rz = log(aTrack.chi2Z());
    float tmp_trk_log_bendchi2 = log(aTrack.stubPtConsistency());
    float tmp_trk_big_invr   = 500*abs(aTrack.rInv());
    float tmp_trk_tanl  = abs(aTrack.tanL());
    float tmp_trk_z0   = abs( aTrack.z0() );


    // fill tensor with track params
    transformed_features.push_back(float(tmp_trk_log_chi2));
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
    transformed_features.push_back(float(tmp_trk_ltot));

    return transformed_features;

    }
}

