#ifndef FeatureTransform_HH
#define FeatureTransform_HH


#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>

namespace FeatureTransform {
  std::vector<float> Transform(TTTrack < Ref_Phase2TrackerDigi_ > aTrack,std::vector<std::string> in_features);
}
#endif

