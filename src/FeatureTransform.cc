#include "L1Trigger/TrackFindingTracklet/interface/FeatureTransform.h"
#include <iostream>
#include <memory>


namespace FeatureTransform {


std::vector<float> Transform(TTTrack < Ref_Phase2TrackerDigi_ > aTrack){

    std::vector<float> transformed_features;


    std::vector<int> hitpattern_binary = {0,0,0,0,0,0,0};
    std::vector<int> hitpattern_expanded_binary = {0,0,0,0,0,0,0,0,0,0,0,0};
    std::vector<float> eta_bins = {0.0,0.2,0.41,0.62,0.9,1.26,1.68,2.08,2.4};

    int hitmap[8][7] = {{0, 1,  2,  3,  4,  5,  12},
                        {0, 1,  2,  3,  4,  5,  12},
                        {0, 1,  2,  3,  4,  5,  12},
                        {0, 1,  2,  3,  4,  5,  12},
                        {0, 1,  2,  3,  4,  5,  12},
                        {0, 1,  2,  6,  7,  8,  9 },
                        {0, 1,  7,  8,  9, 10,  12},
                        {0, 6,  7,  8,  9, 10,  12}};


    for (int i = 6; i >=0; i--){
        int k = aTrack.hitPattern() >> i;
            if (k&1)
                hitpattern_binary[i] = 1;
        }


    float eta = abs(aTrack.eta());
    for (int j=0; j<9; j++)
        {
          if (eta >= eta_bins[j] && eta < eta_bins[j+1])
          {
              for (int k=0; k<6; k++)
                  hitpattern_expanded_binary[hitmap[j][k]] = hitpattern_binary[k];   
          }
        }

    hitpattern_expanded_binary.pop_back();
    int tmp_trk_ltot;

    for (int i=0; i<6; ++i)
    {
      tmp_trk_ltot += hitpattern_expanded_binary[i];
    }

    int tmp_trk_dtot;

    for (int i=6; i<11; ++i)
    {
      tmp_trk_dtot += hitpattern_expanded_binary[i];
    }

    

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
    transformed_features.push_back(float(tmp_trk_dtot+tmp_trk_ltot));
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

